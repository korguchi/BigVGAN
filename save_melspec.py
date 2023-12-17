import math
import os
import random
import torch
import torch.utils.data
import numpy as np
from librosa.util import normalize
from librosa.filters import mel as librosa_mel_fn
from librosa import resample
import soundfile as sf
import pathlib
from tqdm import tqdm
import argparse
from env import AttrDict, build_env
import json
from meldataset import MelDataset, mel_spectrogram, load_wav
from glob import glob

def save_melspec(h, input_wav_dir, output_mel_dir):
    wav_paths = glob(os.path.join(input_wav_dir, '*.wav'))
    for w in tqdm(wav_paths):
        print('Processing: ', w)
        wav = load_wav(w, h.sampling_rate)
        mel = mel_spectrogram(wav,
                              h.n_fft,
                              h.num_mels,
                              h.sampling_rate,
                              h.hop_size,
                              h.win_size,
                              h.fmin,
                              h.fmax)
        mel.save(os.path.join(output_mel_dir, w + '.npy'), mel, allow_pickle=False)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', default='')
    parser.add_argument('--input_wav_dir', type=str, default='JSUT/wav')
    parser.add_argument('--output_mel_dir', type=str, default='JSUT/mel')
    
    a = parser.parse_args()

    with open(a.config) as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)

    save_melspec(h, a.input_wav_dir, a.output_mel_dir)

if __name__ == '__main__':
    main()