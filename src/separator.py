import numpy as np
from scipy.signal import butter, filtfilt
import soundfile as sf
import librosa
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

def create_harmonic_percussive_masks(y):
    """
    Creates soft masks for harmonic and percussive separation.

    Args:
        y: audio time series

    Returns:
        D: original STFT
        harmonic_mask: soft mask for harmonic content
        percussive_mask: soft mask for percussive content
    """
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    
    D = librosa.stft(y)
    D_harmonic = librosa.stft(y_harmonic)
    D_percussive = librosa.stft(y_percussive)
    
    total = np.abs(D_harmonic) + np.abs(D_percussive) + 1e-10
    harmonic_mask = np.abs(D_harmonic) / total
    percussive_mask = np.abs(D_percussive) / total
    
    return D, harmonic_mask, percussive_mask


def create_vocal_mask(harmonic_mask, sr, low_hz=80, high_hz=4000):
    """
    Creates a vocal isolation mask from a harmonic mask.

    Args:
        harmonic_mask: soft harmonic mask
        sr: sample rate
        low_hz: lower vocal frequency bound
        high_hz: upper vocal frequency bound

    Returns:
        vocal_mask: mask isolating vocal frequency range
    """
    freq_bins = librosa.fft_frequencies(sr=sr)
    vocal_range = (freq_bins >= low_hz) & (freq_bins <= high_hz)
    
    vocal_mask = harmonic_mask.copy()
    vocal_mask[~vocal_range, :] = 0
    
    return vocal_mask


def apply_mask(D, mask, sr):
    """
    Applies a mask to a spectrogram and converts back to audio.

    Args:
        D: STFT matrix
        mask: mask to apply
        sr: sample rate

    Returns:
        separated audio as numpy array
    """
    D_masked = D * mask
    return librosa.istft(D_masked)