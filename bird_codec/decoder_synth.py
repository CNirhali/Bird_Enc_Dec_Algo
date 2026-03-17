import numpy as np
import scipy.io.wavfile as wav
import os

class NatureSynthDecoder:
    """
    Decodes a 1KB "Bird footprint" and synthesizes a generative bird song
    representing the original dataset's structural patterns.
    """
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
        # Typical bird song frequencies sit between 2kHz and 8kHz
        self.min_freq = 2000
        self.max_freq = 8000
        
    def _synthesize_bird_call(self, coefficients: np.ndarray, transform_id: int) -> np.ndarray:
        """
        Maps mathematical coefficients to a synthetic generated audio waveform.
        Transform ID determines the "species" or style of the bird call.
        """
        # Duration of the synthesized audio
        duration_sec = 3.0 
        t = np.linspace(0, duration_sec, int(self.sample_rate * duration_sec), endpoint=False)
        audio = np.zeros_like(t)
        
        # 0: DCT (Structured - rhythmic, pure tones)
        # 1: FFT (Periodic - melodic, sweeping frequencies)
        # 2: DWT (Transient/Noisy - sharp chirps and clicks)
        
        # Map our 255 coefficients across time
        num_coeffs = len(coefficients)
        time_steps = np.linspace(0, duration_sec, num_coeffs)
        
        for i, mag in enumerate(coefficients):
            if mag <= 0.01: # Skip negligible magnitudes
                continue
                
            # Map index to frequency (linear mapping to the bird hearing range)
            freq = self.min_freq + (i / num_coeffs) * (self.max_freq - self.min_freq)
            
            # Map time step (when the note occurs)
            start_time = time_steps[i]
            # Duration of the chirp
            chirp_dur = 0.05 + (0.1 * mag)
            
            # Create a time array just for this chirp
            chirp_t = t[(t >= start_time) & (t < start_time + chirp_dur)]
            local_t = chirp_t - start_time
            
            # Envelope to avoid popping (Hann window)
            envelope = np.hanning(len(chirp_t))
            
            if transform_id == 0:
                # DCT: pure rhythmic whistles
                chirp = np.sin(2 * np.pi * freq * local_t)
            elif transform_id == 1:
                # FFT: Melodic frequency sweeps (FM modulation)
                sweep_freq = freq + 1000 * np.sin(2 * np.pi * 5 * local_t)
                chirp = np.sin(2 * np.pi * sweep_freq * local_t)
            else:
                # DWT: Sharp transient clicks and noisy trills
                chirp = np.sin(2 * np.pi * freq * local_t) * np.random.uniform(0.5, 1.0, len(local_t))
            
            # Add to main audio buffer
            audio[(t >= start_time) & (t < start_time + chirp_dur)] += chirp * envelope * mag
            
        return audio

    def decode_to_wav(self, footprint: bytes, output_path: str):
        """
        Parses the 1KB footprint and outputs a synthetic bird song .wav file.
        """
        if len(footprint) != 1024:
            raise ValueError(f"Invalid footprint size. Expected 1024 bytes, got {len(footprint)}.")
            
        header_bytes = footprint[:4]
        coeff_bytes = footprint[4:]
        
        transform_id = np.frombuffer(header_bytes, dtype=np.int32)[0]
        coefficients = np.frombuffer(coeff_bytes, dtype=np.float32)
        
        print(f"Decoding 1KB footprint...")
        print(f"Detected Origin Transform Model: {transform_id}")
        print(f"Synthesizing bird song from {len(coefficients)} data-driven coefficients...")
        
        audio = self._synthesize_bird_call(coefficients, transform_id)
        
        # Normalize audio to 16-bit PCM for WAV compatibility
        max_amp = np.max(np.abs(audio))
        if max_amp > 0:
            audio = audio / max_amp
            
        audio_int16 = np.int16(audio * 32767)
        wav.write(output_path, self.sample_rate, audio_int16)
        print(f"Successfully synthesized native sonification to: {os.path.abspath(output_path)}")
        return audio_int16
