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
from meldataset import MelDataset, load_wav
from glob import glob
import numpy as np
from scipy.signal import stft, hann


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)


def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C


def spectral_normalize(magnitudes):
    output = dynamic_range_compression(magnitudes)
    return output


mel_basis = {}
hann_window = {}


def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    if np.min(y) < -1.:
        print('min value is ', np.min(y))
    if np.max(y) > 1.:
        print('max value is ', np.max(y))

    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)
        mel_basis[str(fmax)] = mel
        hann_window = hann(win_size)

    pad_width = int((n_fft - hop_size) / 2)
    y = np.pad(y, pad_width, mode='reflect')

    _, _, spec = stft(y, fs=sampling_rate, window=hann_window, nperseg=win_size, noverlap=n_fft-hop_size)

    # Compute magnitude
    spec = np.abs(spec)

    # Apply mel scale
    spec = np.dot(mel_basis[str(fmax)], spec)

    # Normalize
    spec = spectral_normalize(spec)

    return spec

def save_melspec(h, input_wav_dir, output_mel_dir):
    os.makedirs(output_mel_dir, exist_ok=True)
    
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
        np.save(os.path.join(output_mel_dir, os.path.basename(w).replace('.wav', '.npy')), mel, allow_pickle=False)


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