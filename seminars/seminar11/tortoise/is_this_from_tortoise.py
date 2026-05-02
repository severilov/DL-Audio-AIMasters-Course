import argparse
import os
import sys

if __package__ in (None, ''):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tortoise.api import classify_audio_clip
from tortoise.utils.audio import load_audio

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clip', type=str, help='Path to an audio clip to classify.', default="../examples/favorite_riding_hood.mp3")
    args = parser.parse_args()

    clip = load_audio(args.clip, 24000)
    clip = clip[:, :220000]
    prob = classify_audio_clip(clip)
    print(f"This classifier thinks there is a {prob*100}% chance that this clip was generated from Tortoise.")
