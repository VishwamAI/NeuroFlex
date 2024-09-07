import numpy as np
from scipy import signal
from typing import List, Dict, Any

class BCIProcessor:
    def __init__(self, sampling_rate: int, num_channels: int):
        self.sampling_rate = sampling_rate
        self.num_channels = num_channels
        self.filters = self._create_filters()

    def _create_filters(self) -> Dict[str, tuple]:
        # Create bandpass filters for different frequency bands
        nyquist = 0.5 * self.sampling_rate
        filters = {
            'delta': signal.butter(4, [0.5, 4], btype='bandpass', fs=self.sampling_rate),
            'theta': signal.butter(4, [4, 8], btype='bandpass', fs=self.sampling_rate),
            'alpha': signal.butter(4, [8, 13], btype='bandpass', fs=self.sampling_rate),
            'beta': signal.butter(4, [13, 30], btype='bandpass', fs=self.sampling_rate),
            'gamma': signal.butter(4, [30, 100], btype='bandpass', fs=self.sampling_rate)
        }
        return filters

    def preprocess(self, raw_data: np.ndarray) -> np.ndarray:
        # Apply basic preprocessing steps
        detrended = signal.detrend(raw_data, axis=0)
        normalized = (detrended - np.mean(detrended, axis=0)) / np.std(detrended, axis=0)
        return normalized

    def apply_filters(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        filtered_data = {}
        for band, (b, a) in self.filters.items():
            filtered_data[band] = signal.filtfilt(b, a, data, axis=0)
        return filtered_data

    def extract_features(self, filtered_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        features = {}
        for band, data in filtered_data.items():
            # Calculate power spectral density
            f, psd = signal.welch(data, fs=self.sampling_rate, nperseg=256, axis=0)
            features[f'{band}_power'] = np.mean(psd, axis=1)
        return features

    def process(self, raw_data: np.ndarray) -> Dict[str, np.ndarray]:
        preprocessed = self.preprocess(raw_data)
        filtered = self.apply_filters(preprocessed)
        features = self.extract_features(filtered)
        return features

# Example usage
if __name__ == "__main__":
    # Simulate some raw EEG data
    sampling_rate = 250  # Hz
    duration = 10  # seconds
    num_channels = 32
    t = np.linspace(0, duration, sampling_rate * duration)
    raw_data = np.random.randn(len(t), num_channels) * 10 + np.sin(2 * np.pi * 10 * t)[:, np.newaxis]

    processor = BCIProcessor(sampling_rate, num_channels)
    features = processor.process(raw_data)

    print("Extracted features:")
    for band, power in features.items():
        print(f"{band}: mean power = {np.mean(power):.2f}")
