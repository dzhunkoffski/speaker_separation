from ss.base import BaseModel
import math

import torch
from torch import nn

def _get_out_seq_len(input_len: int, kernel_size: int, padding: int = 0, dilation: int = 1, stride: int = 1):
    return math.floor((input_len + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)

class SpeechEncoder(nn.Module):
    def __init__(self, sr: int, n_filters: int):
        super().__init__()
        # in ms
        L1 = 2.5 / 1000
        L2 = 10 / 1000
        L3 = 20 / 1000

        self.kernel_size1 = int(L1 * sr)
        self.kernel_size2 = int(L2 * sr)
        self.kernel_size3 = int(L3 * sr)
        self.stride = int((L1 * sr) // 2)

        self.encoder_short = nn.Conv1d(
            in_channels=1, out_channels=n_filters, 
            kernel_size=int(L1 * sr), stride=self.stride
        )
        self.encoder_middle = nn.Conv1d(
            in_channels=1, out_channels=n_filters,
            kernel_size=int(L2 * sr), stride=self.stride,
            padding=(self.kernel_size2 - self.kernel_size1) // 2
        )
        self.encoder_long = nn.Conv1d(
            in_channels=1, out_channels=n_filters,
            kernel_size=int(L3 * sr), stride=self.stride,
            padding=(self.kernel_size3 - self.kernel_size1) // 2
        )
        self.activasion = nn.ReLU()
    
    def forward(self, audio):
        # input: [batch_size X 1 X seq_len]
        # output: [batch_size X embed_dim X seq_len]
        short_features = self.encoder_short(audio)
        middle_features = self.encoder_middle(audio)
        long_features = self.encoder_long(audio)

        short_len = _get_out_seq_len(input_len=audio.size()[-1], kernel_size=self.kernel_size1, stride=self.stride)
        middle_len = _get_out_seq_len(input_len=audio.size()[-1], kernel_size=self.kernel_size2, stride=self.stride)
        long_len = _get_out_seq_len(input_len=audio.size()[-1], kernel_size=self.kernel_size3, stride=self.stride)

        return short_features, middle_features, long_features

class ResNetBlock(nn.Module):
    def __init__(self, speaker_embedding: int):
        super().__init__()

        self.cnn1 = nn.Conv1d(in_channels=speaker_embedding, out_channels=speaker_embedding, kernel_size=1)
        self.batch_norm1 = nn.BatchNorm1d(speaker_embedding)
        self.activasion1 = nn.PReLU()
        self.cnn2 = nn.Conv1d(in_channels=speaker_embedding, out_channels=speaker_embedding, kernel_size=1)
        self.batch_norm2 = nn.BatchNorm1d(speaker_embedding)
        self.activasion2 = nn.PReLU()
        self.max_pool = nn.MaxPool1d(kernel_size=3)

    def forward(self, input):
        output = self.cnn1(input)
        output = self.batch_norm1(output)
        output = self.activasion1(output)

        output = self.cnn2(output)
        output = self.batch_norm2(output)
        output = output + input
        output = self.activasion2(output)
        output = self.max_pool(output)

        return output


class SpeakerEncoder(nn.Module):
    def __init__(self, in_features: int, speaker_embedding: int, O: int, n_resnets: int):
        super().__init__()

        self.norm = nn.LayerNorm(in_features)
        self.cnn1 = nn.Conv1d(in_channels=in_features, out_channels=O, kernel_size=1)
        self.activasion = nn.ReLU()
        self.resnets = nn.ModuleList([ResNetBlock(O) for _ in range(n_resnets)])
        self.cnn2 = nn.Conv1d(in_channels=O, out_channels=speaker_embedding, kernel_size=1)
    
    def forward(self, input):
        # input: [batch_size X embed_dim X seq_len]
        # output: [batch_size X embed_dim]
        input = torch.permute(input, (0, 2, 1))
        input = self.norm(input)
        input = torch.permute(input, (0, 2, 1))
        input = self.cnn1(input)
        for resblock in self.resnets:
            input = resblock(input)
        input = self.cnn2(input)
        output = torch.mean(input, dim=-1)
        return output

class GlobalLayerNorm(nn.Module):
    def __init__(self, normalized_shape: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(self.normalized_shape), requires_grad=True)
        self.beta = nn.Parameter(torch.ones(self.normalized_shape), requires_grad=True)

    def forward(self, input):
        mean = torch.mean(input, dim=list(range(1, len(input.size()))), keepdim=True)
        var = torch.square(torch.std(input, dim=list(range(1, len(input.size()))), keepdim=True))
        normalized_input = (input - mean) / torch.sqrt(var + self.eps)
        normalized_input = torch.permute(normalized_input, (0, 2, 1))
        broadcasted_gamma = torch.broadcast_to(self.gamma, normalized_input.size())
        broadcasted_beta = torch.broadcast_to(self.beta, normalized_input.size())
        output = normalized_input * broadcasted_gamma + broadcasted_beta
        output = torch.permute(output, (0, 2, 1))
        return output


    
class TCN(nn.Module):
    def __init__(self, in_features: int, embed_dim: int, dilation: int, Q: int, P: int):
        super().__init__()

        self.embed_dim = embed_dim
        self.cnn1 = nn.Conv1d(
            in_channels=in_features+embed_dim, out_channels=P,
            kernel_size=1
        )
        self.activasion1 = nn.PReLU()
        self.gLN1 = GlobalLayerNorm(normalized_shape=P)
        self.dconv = nn.Conv1d(
            in_channels=P, out_channels=P, 
            kernel_size=Q, groups=P, dilation=dilation,
            padding=(dilation * (Q-1)) // 2
        )
        self.activasion2 = nn.PReLU()
        self.gLN2 = GlobalLayerNorm(normalized_shape=P)
        self.cnn2 = nn.Conv1d(
            in_channels=P, out_channels=in_features,
            kernel_size=1
        )
    
    def forward(self, input, speaker_embedding = None):
        # [x]
        output = input
        if self.embed_dim != 0:
            speaker_embedding = speaker_embedding.unsqueeze(-1).repeat(1,1,input.size()[-1])
            output = torch.cat((output, speaker_embedding), 1)
        output = self.cnn1(output)
        output = self.activasion1(output)
        output = self.gLN1(output)
        output = self.dconv(output)
        output = self.activasion2(output)
        output = self.gLN2(output)
        output = self.cnn2(output)
        output += input
        return output
        
class StackedTCN(nn.Module):
    def __init__(self, n_blocks: int, O: int, Q: int, P: int, embed_dim: int):
        super().__init__()

        self.n_blocks = n_blocks
        self.tcns = nn.ModuleList([
            TCN(in_features=O, embed_dim=embed_dim, dilation=2**0, Q=Q, P=P)
        ])
        for i in range(1, n_blocks):
            self.tcns.append(
                TCN(in_features=O, embed_dim=0, dilation=2**i, Q=Q, P=P)
            )

    def forward(self, input, speaker_embedding):
        output = self.tcns[0](input, speaker_embedding)
        for i in range(1, self.n_blocks):
            output = self.tcns[i](output)
        return output
    
class SpeakerExtractor(nn.Module):
    def __init__(
            self, in_features: int, 
            O: int, Q: int, P: int, embed_dim: int,
            n_tcn_stacks: int, n_tcn_blocks_in_stack: int 
            ):
        super().__init__()

        self.n_stacks = n_tcn_stacks
        self.norm = nn.LayerNorm(in_features)
        self.cnn1 = nn.Conv1d(in_channels=in_features, out_channels=O, kernel_size=1)
        self.tcn_stacks = nn.ModuleList([StackedTCN(n_blocks=n_tcn_blocks_in_stack, O=O, Q=Q, P=P, embed_dim=embed_dim) for _ in range(n_tcn_stacks)])
        self.cnn_short = nn.Conv1d(in_channels=O, out_channels=in_features // 3, kernel_size=1)
        self.cnn_middle = nn.Conv1d(in_channels=O, out_channels=in_features // 3, kernel_size=1)
        self.cnn_long = nn.Conv1d(in_channels=O, out_channels=in_features // 3, kernel_size=1)
        self.activasion = nn.ReLU()
    
    def forward(self, input, speaker_embedding):
        output = torch.permute(input, (0, 2, 1))
        output = self.norm(output)
        output = torch.permute(output, (0, 2, 1))
        output = self.cnn1(output)
        for i in range(self.n_stacks):
            output = self.tcn_stacks[i](output, speaker_embedding)
        mask_short = self.activasion(
            self.cnn_short(output)
        )
        mask_middle = self.activasion(
            self.cnn_middle(output)
        )
        mask_long = self.activasion(
            self.cnn_long(output)
        )
        return mask_short, mask_middle, mask_long

class SpeakerClassifier(nn.Module):
    def __init__(self, speaker_embed_dim: int, n_speakers: int):
        super().__init__()
        self.linear = nn.Linear(speaker_embed_dim, n_speakers)
    
    def forward(self, x):
        x = self.linear(x)
        return x


class SpexPlus(BaseModel):
    def __init__(
            self, sr: int, n_encoder_filters: int, speaker_embed_dim: int, n_resnets: int, 
            O: int, Q: int, P: int, n_tcn_stacks: int, n_tcn_blocks_in_stack: int, n_speakers: int = 0, use_speaker_class = False
            ):
        """
        :param n_encoder_filters: out_channels value for CNNs in speech encoder
        :param speaker_embed_dim: size of the speaker embedding
        :param n_resnets: numnber of resnet blocks in speaker encoder
        :param O: number of filters in CNN before the TCN block
        :param Q: kernel size for d-conv in TCN
        :param P: number of channels in 1x1 conv in TCN
        """ 
        super().__init__()

        L1 = 2.5 / 1000
        L2 = 10 / 1000
        L3 = 20 / 1000

        self.speech_encoder = SpeechEncoder(sr, n_encoder_filters)
        self.speaker_encoder = SpeakerEncoder(
            in_features=n_encoder_filters * 3, 
            O=O,
            speaker_embedding=speaker_embed_dim, 
            n_resnets=n_resnets
        )
        self.speaker_extractor = SpeakerExtractor(
            in_features=n_encoder_filters * 3,
            O=O, Q=Q, P=P,
            embed_dim=speaker_embed_dim,
            n_tcn_stacks=n_tcn_stacks, n_tcn_blocks_in_stack=n_tcn_blocks_in_stack
        )

        # FIXME: maybe should apply different kernel size
        self.decoder_short = nn.ConvTranspose1d(
            in_channels=n_encoder_filters, out_channels=1, 
            kernel_size=int(L1 * sr), stride=int(L1 * sr // 2)
        )
        self.decoder_middle = nn.ConvTranspose1d(
            in_channels=n_encoder_filters, out_channels=1,
            kernel_size=int(L2 * sr), stride=int(L1 * sr // 2),
            padding=(int(L2 * sr) - int(L1 * sr)) // 2
        )
        self.decoder_long = nn.ConvTranspose1d(
            in_channels=n_encoder_filters, out_channels=1,
            kernel_size=int(L3 * sr), stride=int(L1 * sr // 2),
            padding=(int(L3 * sr) - int(L1 * sr)) // 2
        )
        self.activasion = nn.ReLU()

        self.n_speakers = n_speakers
        if n_speakers > 0 and use_speaker_class:
            self.speaker_clf = SpeakerClassifier(
                speaker_embed_dim=speaker_embed_dim, n_speakers=n_speakers
            )
        else:
            self.n_speakers = 0


    def forward(self, mix, reference, is_train: bool, **batch):
        # mix = input['mix']
        # ref = input['reference']

        short_features, middle_features, long_features = self.speech_encoder(mix)
        mix_features = torch.cat((short_features, middle_features, long_features), dim=1)
        mix_features = self.activasion(mix_features)

        speaker_embedding1, speaker_embedding2, speaker_embedding3 = self.speech_encoder(reference)
        speaker_embedding = torch.cat((speaker_embedding1, speaker_embedding2, speaker_embedding3), dim=1)
        speaker_embedding = self.activasion(speaker_embedding)
        speaker_embedding = self.speaker_encoder(speaker_embedding)

        speaker_logits = None
        if self.n_speakers > 0 and is_train:
            speaker_logits = self.speaker_clf(speaker_embedding)


        mask_short, mask_middle, mask_long = self.speaker_extractor(mix_features, speaker_embedding)
        short_features = torch.mul(short_features, mask_short)
        middle_features = torch.mul(middle_features, mask_middle)
        long_features = torch.mul(long_features, mask_long)

        short_features = self.decoder_short(short_features)
        middle_features = self.decoder_middle(middle_features)
        long_features = self.decoder_long(long_features)

        # FIXME: different lengths
        print('mix:', mix.size())
        print('short:', short_features.size())
        print('middle:', middle_features.size())
        print('long:', long_features.size())

        short_features = nn.functional.pad(short_features, pad=(0, mix.size()[-1] - short_features.size()[-1]), mode='constant', value=0)
        middle_features = nn.functional.pad(middle_features, pad=(0, mix.size()[-1] - middle_features.size()[-1]), mode='constant', value=0)
        long_features = nn.functional.pad(long_features, pad=(0, mix.size()[-1] - long_features.size()[-1]), mode='constant', value=0)

        return {'s1': short_features, 's2': middle_features, 's3': long_features, 'sp_logits': speaker_logits}
