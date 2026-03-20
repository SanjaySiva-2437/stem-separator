import numpy as np
from scipy.signal import butter, filtfilt
import soundfile as sf
import os


def butter_filter(y, sr, filter_type, lowcut=None, highcut=None, order=5):
    """
    Applies a Butterworth filter to an audio signal.

    Args:
        y: audio time series
        sr: sample rate
        filter_type: 'low', 'high' or 'band'
        lowcut: lower frequency cutoff (Hz)
        highcut: upper frequency cutoff (Hz)
        order: filter sharpness

    Returns:
        filtered audio as numpy array
    """
    nyquist = sr / 2

    if filter_type == 'low':
        cutoff = highcut / nyquist
        b, a = butter(order, cutoff, btype='low')
    elif filter_type == 'high':
        cutoff = lowcut / nyquist
        b, a = butter(order, cutoff, btype='high')
    elif filter_type == 'band':
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')

    return filtfilt(b, a, y)


def separate_by_frequency(y, sr, bass_cutoff=300, mid_cutoff=3000):
    """
    Separates audio into bass, mids and treble bands.

    Args:
        y: audio time series
        sr: sample rate
        bass_cutoff: upper frequency limit for bass (Hz)
        mid_cutoff: upper frequency limit for mids (Hz)

    Returns:
        bass, mids, treble as numpy arrays
    """
    bass = butter_filter(y, sr, 'low', highcut=bass_cutoff)
    mids = butter_filter(y, sr, 'band', lowcut=bass_cutoff, highcut=mid_cutoff)
    treble = butter_filter(y, sr, 'high', lowcut=mid_cutoff)

    return bass, mids, treble


def save_stems(stems, sr, output_dir):
    """
    Saves separated stems as WAV files.

    Args:
        stems: dictionary of stem name to audio array
        sr: sample rate
        output_dir: folder to save files in
    """
    os.makedirs(output_dir, exist_ok=True)

    for name, audio in stems.items():
        path = os.path.join(output_dir, f"{name}.wav")
        sf.write(path, audio, sr)
        print(f"Saved: {path}")