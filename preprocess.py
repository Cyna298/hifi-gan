import glob
import os
from pathlib import Path
from tqdm import tqdm

import numpy as np
from TTS.utils.audio import AudioProcessor


def preprocess_wav_files(out_path, config, ap):
    os.makedirs(os.path.join(out_path, "quant"), exist_ok=True)
    os.makedirs(os.path.join(out_path, "mel"), exist_ok=True)
    wav_files = find_wav_files(config.data_path)
    for path in tqdm(wav_files):
        wav_name = Path(path).stem
        quant_path = os.path.join(out_path, "quant", wav_name + ".npy")
        mel_path = os.path.join(out_path, "mel", wav_name + ".npy")
        y = ap.load_wav(path)
        mel = ap.melspectrogram(y)
        np.save(mel_path, mel)
        if isinstance(config.mode, int):
            quant = (
                ap.mulaw_encode(y, qc=config.mode)
                if config.mulaw
                else ap.quantize(y, bits=config.mode)
            )
            np.save(quant_path, quant)


def find_wav_files(data_path):
    wav_paths = glob.glob(os.path.join(data_path, "**", "*.wav"), recursive=True)
    return wav_paths

