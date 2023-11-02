import os
from glob import glob

class LibriSpeechSpeakerFiles:
    def __init__(self, speaker_id: int, audios_dir: str, audio_template: str = "*-norm.wav"):
        self.id = speaker_id
        self.files = []
        self.audio_template = audio_template
        self.files = self.find_files_by_worker(audios_dir)
    
    def find_files_by_worker(self, audios_dir: str):
        speaker_dir = os.path.join(audios_dir, self.id)
        chapter_dirs = os.scandir(speaker_dir)
        files = []
        for chapter_dir in chapter_dirs:
            files = files + [file for file in glob(os.path.join(speaker_dir,chapter_dir.name)+"/"+self.audio_template)]
        return files