import torch
import librosa
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split


def audio_to_mel(audio, sr, n_mels=128, n_fft=2048, hop_length=512):
    """Converts stereo audio to mel spectrogram."""
    audio_mono = audio.mean(axis=1)
    M = librosa.feature.melspectrogram(
        y=audio_mono, sr=sr, n_mels=n_mels,
        n_fft=n_fft, hop_length=hop_length
    )
    return librosa.power_to_db(M, ref=np.max)


class MusdbDataset(Dataset):
    def __init__(self, db, n_mels=128, n_fft=2048, hop_length=512):
        self.db = db
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length

    def __len__(self):
        return len(self.db)

    def __getitem__(self, idx):
        track = self.db[idx]
        sr = track.rate

        mixture = audio_to_mel(track.audio, sr, self.n_mels, self.n_fft, self.hop_length)
        vocals = audio_to_mel(track.sources['vocals'].audio, sr, self.n_mels, self.n_fft, self.hop_length)
        drums = audio_to_mel(track.sources['drums'].audio, sr, self.n_mels, self.n_fft, self.hop_length)
        bass = audio_to_mel(track.sources['bass'].audio, sr, self.n_mels, self.n_fft, self.hop_length)
        other = audio_to_mel(track.sources['other'].audio, sr, self.n_mels, self.n_fft, self.hop_length)

        mixture = torch.tensor(mixture, dtype=torch.float32).unsqueeze(0)
        vocals = torch.tensor(vocals, dtype=torch.float32).unsqueeze(0)
        drums = torch.tensor(drums, dtype=torch.float32).unsqueeze(0)
        bass = torch.tensor(bass, dtype=torch.float32).unsqueeze(0)
        other = torch.tensor(other, dtype=torch.float32).unsqueeze(0)

        return mixture, torch.cat([vocals, drums, bass, other], dim=0)


def get_dataloaders(db, batch_size=8, train_split=0.8):
    """Creates train and validation DataLoaders."""
    dataset = MusdbDataset(db)
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader