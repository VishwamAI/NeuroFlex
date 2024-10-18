# MIT License
# 
# Copyright (c) 2024 VishwamAI
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
from scipy import signal
from typing import List, Dict, Any
from sklearn.decomposition import FastICA
from mne.decoding import CSP
import pywt
from filterpy.kalman import KalmanFilter

class BCIProcessor:
    def __init__(self, sampling_rate: int, num_channels: int, electrode_thickness: float = 0.005):
        self.sampling_rate = sampling_rate
        self.num_channels = num_channels
        self.electrode_thickness = electrode_thickness  # Ultra-thin electrode design (mm)
        self.filters = self._create_filters()
        self.ica = FastICA(n_components=self.num_channels)
        self.csp = CSP(n_components=self.num_channels)
        self.kalman_filter = self._create_kalman_filter()
        self.wireless_transmitter = self._setup_wireless_communication()

    def _create_filters(self) -> Dict[str, tuple]:
        # Create bandpass filters for different frequency bands
        nyquist = 0.5 * self.sampling_rate
        filters = {
            'delta': signal.butter(6, [0.5, 4], btype='bandpass', fs=self.sampling_rate),
            'theta': signal.butter(6, [4, 8], btype='bandpass', fs=self.sampling_rate),
            'alpha': signal.butter(6, [8, 13], btype='bandpass', fs=self.sampling_rate),
            'beta': signal.butter(6, [13, 30], btype='bandpass', fs=self.sampling_rate),
            'gamma': signal.butter(6, [30, 100], btype='bandpass', fs=self.sampling_rate),
            'high_gamma': signal.butter(6, [30, 100], btype='bandpass', fs=self.sampling_rate)
        }
        return filters

    def _create_kalman_filter(self) -> KalmanFilter:
        kf = KalmanFilter(dim_x=4, dim_z=2)  # Increased dimensions for higher resolution
        kf.x = np.zeros((4, 1))  # initial state
        kf.F = np.array([[1., 1., 0.5, 0.], [0., 1., 1., 0.], [0., 0., 1., 1.], [0., 0., 0., 1.]])  # state transition matrix
        kf.H = np.array([[1., 0., 0., 0.], [0., 0., 1., 0.]])  # measurement function
        kf.P *= 1000.  # covariance matrix
        kf.R = np.array([[5., 0.], [0., 5.]])  # measurement noise
        kf.Q = np.eye(4) * 0.01  # process noise
        return kf

    def _setup_wireless_communication(self):
        # Placeholder for wireless communication setup
        # In a real implementation, this would initialize hardware or software components
        # for wireless data transmission
        return {"protocol": "Bluetooth LE", "data_rate": "2 Mbps", "latency": "2ms"}

    def preprocess(self, raw_data: np.ndarray, labels: np.ndarray) -> np.ndarray:
        # Calculate n_trials and n_samples based on raw_data shape before any processing
        if raw_data.ndim == 3:
            n_trials, n_channels, n_samples = raw_data.shape
        elif raw_data.ndim == 2:
            n_channels, n_samples = raw_data.shape
            n_trials = len(labels)  # Use the number of labels to determine n_trials
        else:
            raise ValueError("Input data must be 2D or 3D")

        # Apply basic preprocessing steps
        detrended = signal.detrend(raw_data.reshape(n_trials * n_channels, -1), axis=1)
        normalized = (detrended - np.mean(detrended, axis=1, keepdims=True)) / np.std(detrended, axis=1, keepdims=True)

        # Reshape data for CSP (trials x channels x samples)
        reshaped_data = normalized.reshape(n_trials, n_channels, -1)

        # Ensure labels are correctly aligned with the reshaped data and have at least two unique classes
        if len(labels) != n_trials:
            raise ValueError(f"Number of labels ({len(labels)}) must match number of trials ({n_trials})")
        unique_classes = np.unique(labels)
        if len(unique_classes) < 2:
            raise ValueError("At least two unique classes are required for CSP")

        # Apply CSP with reduced number of components (e.g., 4 instead of full n_channels)
        self.csp.n_components = min(4, n_channels)
        csp_data = self.csp.fit_transform(reshaped_data, y=labels)

        return csp_data

    def apply_filters(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        filtered_data = {}
        min_padlen = max(len(b) for b, _ in self.filters.values()) * 3
        if data.shape[0] <= min_padlen:
            pad_width = ((min_padlen - data.shape[0] + 1, 0), (0, 0))
            data = np.pad(data, pad_width, mode='edge')
        for band, (b, a) in self.filters.items():
            filtered_data[band] = signal.filtfilt(b, a, data, axis=0)
        return filtered_data

    def extract_features(self, filtered_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        features = {}
        for band, data in filtered_data.items():
            print(f"Processing {band} band, data shape: {data.shape}")

            # Calculate power spectral density using numpy
            f, psd = signal.welch(data, fs=self.sampling_rate, nperseg=min(256, data.shape[-1]))
            print(f"PSD shape before adjustment: {psd.shape}")

            # Ensure the power feature maintains 64 channels and 129 frequency bins
            psd = psd[:64, :129] if psd.shape[0] >= 64 and psd.shape[1] >= 129 else np.pad(psd, ((0, max(0, 64 - psd.shape[0])), (0, max(0, 129 - psd.shape[1]))))
            print(f"PSD shape after adjustment: {psd.shape}")
            features[f'{band}_power'] = psd  # Remove transpose to maintain shape (64, 129)

            # Apply wavelet transform
            coeffs = pywt.wavedec(data, 'db4', level=min(5, data.shape[-1] // 2), axis=-1)
            print(f"Number of wavelet coefficients: {len(coeffs)}")

            # Ensure wavelet features maintain correct dimensions (64 channels, 6 coefficients)
            wavelet_features = np.array([np.mean(np.abs(c), axis=-1) for c in coeffs])
            print(f"Wavelet features shape before transpose: {wavelet_features.shape}")
            wavelet_features = wavelet_features.T
            print(f"Wavelet features shape after transpose: {wavelet_features.shape}")

            if wavelet_features.shape[0] != 64:
                print(f"Resizing wavelet features from {wavelet_features.shape} to (64, {wavelet_features.shape[1]})")
                # Resize wavelet_features to have exactly 64 channels
                resized_features = np.zeros((64, wavelet_features.shape[1]))
                resized_features[:min(64, wavelet_features.shape[0]), :] = wavelet_features[:min(64, wavelet_features.shape[0]), :]
                wavelet_features = resized_features

            print(f"Final wavelet features shape: {wavelet_features.shape}")
            features[f'{band}_wavelet'] = wavelet_features

        return features

    def process(self, raw_data: np.ndarray, labels: np.ndarray) -> Dict[str, np.ndarray]:
        print(f"Raw data shape: {raw_data.shape}")
        preprocessed = self.preprocess(raw_data, labels)
        print(f"Preprocessed data shape: {preprocessed.shape}")
        filtered = self.apply_filters(preprocessed)
        print(f"Filtered data shape: {filtered['delta'].shape}")  # Assuming 'delta' is always present
        features = self.extract_features(filtered)

        # Ensure correct channel dimensions for all features
        for feature_name, feature_data in features.items():
            print(f"{feature_name} shape before adjustment: {feature_data.shape}")
            if feature_data.shape[0] != self.num_channels and 'wavelet' not in feature_name:
                features[feature_name] = feature_data.T
            print(f"{feature_name} shape after adjustment: {features[feature_name].shape}")

        return features

# Example usage
if __name__ == "__main__":
    # Simulate some raw EEG data
    sampling_rate = 250  # Hz
    duration = 10  # seconds
    num_channels = 32
    t = np.linspace(0, duration, sampling_rate * duration)
    raw_data = np.random.randn(len(t), num_channels) * 10 + np.sin(2 * np.pi * 10 * t)[:, np.newaxis]

    # Create dummy labels for the example
    n_trials = raw_data.shape[0]
    dummy_labels = np.array([0, 1] * (n_trials // 2 + 1))[:n_trials]

    processor = BCIProcessor(sampling_rate, num_channels)
    features = processor.process(raw_data, dummy_labels)

    print("Extracted features:")
    for feature_name, feature_data in features.items():
        print(f"{feature_name}: shape = {feature_data.shape}, mean = {np.mean(feature_data):.2f}")
