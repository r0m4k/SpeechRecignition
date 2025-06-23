import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import os
from datetime import timedelta
import textwrap
import subprocess

class AudioRecognition:
    
    def __init__(self):
        self.file = None
        self.text = None
        self.model = self.load_model()

    def load_model(self):
        """Method that loads the speech recognition model and saves it to the self.model attribute"""

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        model_id = "model/whisper-large-v3-turbo"
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id)

        pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
            )   
        return pipe
    
    def get_text(self, filepath):
        """Method that takes a path to an audio file and returns the recognized text"""

        self.file = filepath
        self.text = self.model(self.file, return_timestamps=True, generate_kwargs={"language": "english"})
        return self.text
    
    def audio_to_file(self, audiofile):
        """Method that takes a path to an audio file and writes the recognized text to audio_to_text.txt file"""

        with open("audio_to_text_one.txt", "w") as file:
            result = self.get_text(audiofile)
            wrapped_text = textwrap.fill(result["text"], width=80)
            file.write(wrapped_text)
    
    def split_audiofile(self, audiofile, duration=30):
        """Method that takes a path to an audio file and splits it into audio files with duration from the duration argument.
        Necessary for optimizing recognition of long audio files"""

        if not os.path.exists('samples'):
            os.makedirs(f'samples')

        output_file = os.path.join(f'samples', "sample_%03d.wav")
        command = f"ffmpeg -i {audiofile} -f segment -segment_time {duration} -c copy {output_file}"
        subprocess.run(command, shell=True)
    
    def large_audio_to_file(self, audiofile, duration=30):
        """Method that takes a path to a large audio file and writes the recognized text to audio_to_text.txt file."""

        self.file = audiofile
        self.split_audiofile(self.file)

        with open("audio_to_text.txt", "w") as file:
            # initialize time to add timestamps every 30 seconds to the file
            time = timedelta(seconds=0)
            audio_files = [audio_file for audio_file in list(os.listdir('samples')) if audio_file.endswith(".wav")]
            audio_files.sort()
            for audio in audio_files:
                print(audio)
                result = self.model(os.path.join('samples', audio), return_timestamps=True, generate_kwargs={"language": "english"})
                file.write(f'[{time} - {time + timedelta(seconds=duration)}] \n')
                wrapped_text = textwrap.fill(result["text"], width=80)  # specify the maximum width of a line
                file.write(wrapped_text + "\n")
                time += timedelta(seconds=30)

    def mp4_to_mp3(self, mp4file):
        """Method that takes a path to an mp4 file and converts it to an mp3 file"""

        mp3file = mp4file.replace(".mp4", ".mp3")
        command = f"ffmpeg -i {mp4file} -vn -ar 44100 -ac 2 -ab 192k -f mp3 {mp3file}"
        subprocess.run(command, shell=True)
        return mp3file
