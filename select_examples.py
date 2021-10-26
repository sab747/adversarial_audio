'''
select_examples.py

Created by Sabrina Jain
'''
import os, sys
from pydub import AudioSegment
import pandas as pd

def select_clips (clips, label, num_samples):
    clips_for_label = clips[clips["sentence"] == label]
    sampled_clips = clips_for_label.sample(n = num_samples)
    return sampled_clips

def convert_clips (sampled_clips, clips_label):
    for sampled_clips, clip in sampled_clips.iterrows():
        clip_path = clip["path"]
        convert_mp3_to_wav(clips_label, clip_path)

def convert_mp3_to_wav (clips_label, clip_path):
    sound = AudioSegment.from_mp3("./data/cv-corpus-7.0-singleword/en/clips/" + clip_path)
    sound.export("./samples/" + clips_label + "/" + clip_path[:-4] + ".wav", format="wav")
    
if __name__ == '__main__':
    
    sys.path.append('/path/to/ffmpeg')
    print(os.environ['PATH'])
    labels = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "yes", "no"]

    clips = pd.read_csv("./data/cv-corpus-7.0-singleword/en/validated.tsv", sep='\t')
    for label in labels:
        os.mkdir("./samples/" + label)
        clips_for_label = select_clips(clips, label, 50)
        convert_clips(clips_for_label, label)