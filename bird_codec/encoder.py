import numpy as np
import pywt
from scipy.fftpack import dct

class NatureTransformEncoder:
    """
    Dynamically applies spectral transforms to massive datasets to extract
    their dominant structural patterns, compressing them into a 1KB "Bird footprint".
    """
    def __init__(self, target_size_bytes=1024):
        self.target_size_bytes = target_size_bytes
        # 1 KB = 256 float32 values. We reserve 4 bytes for metadata (transform type).
        self.num_coefficients = (target_size_bytes - 4) // 4
        
    def _calculate_entropy(self, data: np.ndarray) -> float:
        """Calculates Shannon entropy to dynamically choose the best transform."""
        if len(data) == 0:
            return 0.0
            
        # Create a histogram to estimate probabilities
        hist, _ = np.histogram(data, bins=256, density=True)
        # Filter out zero probabilities safely
        p = hist[hist > 0]
        return -np.sum(p * np.log2(p))

    def _apply_fft(self, data: np.ndarray):
        """Extracts repeating periodic patterns using FFT."""
        # Compute magnitude spectrum
        spectrum = np.abs(np.fft.rfft(data))
        # Get indices of the most dominant frequencies
        top_indices = np.argsort(spectrum)[-self.num_coefficients:]
        # Extract the dominant values
        dominant_values = spectrum[top_indices]
        return 0, dominant_values

    def _apply_dct(self, data: np.ndarray):
        """Extracts energy concentration using Discrete Cosine Transform."""
        spectrum = np.abs(dct(data, type=2, norm='ortho'))
        top_indices = np.argsort(spectrum)[-self.num_coefficients:]
        dominant_values = spectrum[top_indices]
        return 1, dominant_values

    def _apply_dwt(self, data: np.ndarray):
        """Extracts transient spikes and localized features using Discrete Wavelet Transform."""
        # Use a smooth wavelet suitable for audio (Daubechies)
        coeffs = pywt.wavedec(data, 'db4', level=5)
        # Flatten all detail and approximation coefficients
        flat_coeffs = np.concatenate([np.abs(c) for c in coeffs])
        
        # Avoid out of bounds if data is incredibly small
        if len(flat_coeffs) < self.num_coefficients:
            padded = np.zeros(self.num_coefficients)
            padded[:len(flat_coeffs)] = flat_coeffs
            return 2, padded
            
        top_indices = np.argsort(flat_coeffs)[-self.num_coefficients:]
        dominant_values = flat_coeffs[top_indices]
        return 2, dominant_values

    def encode(self, data: np.ndarray) -> bytes:
        """
        Compresses an arbitrarily large dataset into a 1KB footprint by analyzing
        its structural entropy and dynamically picking a mathematical transform.
        """
        if len(data) == 0:
            raise ValueError("Input data cannot be empty.")
            
        print(f"Analyzing dataset of size: {len(data) * data.itemsize / (1024**2):.2f} MB")
        
        entropy = self._calculate_entropy(data)
        print(f"Calculated Data Entropy: {entropy:.2f} bits/symbol")

        # Dynamic Selection Logic
        # - Low entropy (highly structured): DCT captures energy best
        # - Medium entropy (periodic): FFT captures frequencies
        # - High entropy (noisy/transient): DWT captures localized spikes
        if entropy < 3.0:
            print("Selected Transform: Discrete Cosine Transform (DCT) [Highly Structured Data]")
            transform_id, coeffs = self._apply_dct(data)
        elif entropy < 7.0:
            print("Selected Transform: Fast Fourier Transform (FFT) [Periodic Data]")
            transform_id, coeffs = self._apply_fft(data)
        else:
            print("Selected Transform: Discrete Wavelet Transform (DWT) [High Entropy/Transient Data]")
            transform_id, coeffs = self._apply_dwt(data)
            
        # Normalize coefficients to fit nicely in the audio domain [-1.0, 1.0]
        max_val = np.max(np.abs(coeffs))
        if max_val > 0:
            coeffs = coeffs / max_val
            
        # Pack precisely to 1KB:
        # [4 bytes int32: Transform ID] + [1020 bytes: 255 float32 coefficients]
        
        # Explicitly take 255 elements to match exactly 1020 bytes
        packed_coeffs = coeffs[-255:].astype(np.float32)
        
        # In case the input data was too small to provide 255 coefficients
        if len(packed_coeffs) < 255:
            padded = np.zeros(255, dtype=np.float32)
            padded[:len(packed_coeffs)] = packed_coeffs
            packed_coeffs = padded
            
        header = np.array([transform_id], dtype=np.int32)
        
        # 4 + (255 * 4) = 1024 bytes
        footprint = header.tobytes() + packed_coeffs.tobytes()
        
        assert len(footprint) == 1024, f"Footprint size is {len(footprint)}, expected 1024"
        return footprint
