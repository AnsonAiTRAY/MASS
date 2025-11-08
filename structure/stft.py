import torch
from torch import nn
import numpy as np

class STFT(nn.Module):
    def __init__(self, n_fft, hop_length, win_length, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nfft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = torch.hamming_window(self.win_length, periodic=False)

    def forward(self, eeg):
        if isinstance(eeg, np.ndarray):
            eeg = torch.from_numpy(eeg).float().unsqueeze(1)
        batch_size, _, signal_length = eeg.shape
        assert signal_length >= self.win_length, "Signal length must be at least as long as win_length"
        # 手动分段和加窗操作
        segments = eeg.unfold(-1, self.win_length, self.hop_length)
        segments = segments * self.window.to(eeg.device)  # 再次加窗（如必要）
        # 进行FFT操作
        stft_result = torch.fft.fft(segments, n=self.nfft, dim=-1)[:, :, :, :self.nfft // 2 + 1]
        # 计算结果
        stft_result = stft_result.abs().transpose(-1, -2)
        magnitude_db = 20 * torch.log10(stft_result + 1e-8).transpose(-1, -2).squeeze(1)  # 加1e-8以避免log(0)
        band1 = magnitude_db[:, :, 1:10]
        band2 = magnitude_db[:, :, 10:20]
        band3 = magnitude_db[:, :, 20:31]
        band4 = magnitude_db[:, :, 31:77]
        band5 = magnitude_db[:, :, 77:129]
        return band1, band2, band3, band4, band5, magnitude_db[:, :, 1:129]


# stft = STFT(n_fft=256, hop_length=100, win_length=100)
# file_dir = '../../EEG-MAE/dataset/EDF20/mat/SC4001.mat'
# array_data, _ = load_dataset(file_dir, 'eeg', 'label')
# array_data = list(array_data)[0]
# signal = torch.tensor(array_data).float().unsqueeze(0).unsqueeze(0)
# print(signal.shape)
# band1, band2, band3, band4, band5, magnitude_db = stft(signal)
# print(band1.shape, band2.shape, band3.shape, band4.shape, band5.shape, magnitude_db.shape)
