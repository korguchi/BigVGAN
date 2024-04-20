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
from meldataset import MelDataset, load_wav, mel_spectrogram
from glob import glob
import numpy as np

def save_melspec(h, input_wavs_dir, output_mels_dir):
    os.makedirs(output_mels_dir, exist_ok=True)
    wav_paths = glob(os.path.join(input_wavs_dir, '**/*.wav'), recursive=True)
    for w in tqdm(wav_paths):
        audio = load_wav(w, h.sampling_rate)
        audio = torch.FloatTensor(audio)
        audio = audio.unsqueeze(0)
        mel = mel_spectrogram(audio,
                              h.n_fft,
                              h.num_mels,
                              h.sampling_rate,
                              h.hop_size,
                              h.win_size,
                              h.fmin,
                              h.fmax)
        mel = mel.to('cpu').detach().numpy().copy()
        mel_dir = os.path.join(output_mels_dir, os.path.dirname(w).replace('wav/', ''))
        os.makedirs(mel_dir, exist_ok=True)
        np.save(os.path.join(mel_dir, os.path.basename(w).replace('.wav', '.npy')), mel, allow_pickle=False)
        # np.save(os.path.join(output_mels_dir, w.replace('wav/', '').replace('.wav', '.npy')), mel, allow_pickle=False)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', default='')
    parser.add_argument('--input_wavs_dir', type=str, default='wav')
    parser.add_argument('--output_mels_dir', type=str, default='mel')
    
    a = parser.parse_args()

    with open(a.config) as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)

    save_melspec(h, a.input_wavs_dir, a.output_mels_dir)

if __name__ == '__main__':
    main()
