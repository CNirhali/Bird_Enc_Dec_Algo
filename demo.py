import os
import numpy as np
from bird_codec.encoder import NatureTransformEncoder
from bird_codec.decoder_synth import NatureSynthDecoder

def generate_dummy_data(scenario: str, size_mb: float = 1.0) -> np.ndarray:
    """Generates large numerical dataset based on a scenario to test dynamic transforms."""
    num_elements = int(size_mb * (1024**2) / 4) # 4 bytes per float32
    
    print(f"\n[{scenario.upper()}] Generating {size_mb} MB of data...")
    if scenario == "structured":
        # Low Entropy: highly repetitive structured patterns
        t = np.linspace(0, 100 * np.pi, num_elements)
        data = np.sin(t) + np.cos(3 * t)
    elif scenario == "periodic":
        # Medium Entropy: complex periodic waves
        t = np.linspace(0, 1000 * np.pi, num_elements)
        data = np.sin(t) * np.exp(-0.001 * t)
    elif scenario == "transient":
        # High Entropy: random noise mimicking transient spikes
        data = np.random.randn(num_elements)
    else:
        raise ValueError("Unknown scenario.")
        
    return data.astype(np.float32)

def run_demo():
    print("=" * 60)
    print("BIRD SONIFICATION & GENERATIVE COMPRESSION PIPELINE DEMO")
    print("=" * 60)

    encoder = NatureTransformEncoder(target_size_bytes=1024)
    decoder = NatureSynthDecoder(sample_rate=44100)
    
    scenarios = ["structured", "periodic", "transient"]
    
    for idx, scenario in enumerate(scenarios):
        # 1. Generate massive dummy dataset
        input_data = generate_dummy_data(scenario, size_mb=10.0) # 10 MB each for the demo
        
        # 2. Encode to 1KB footprint
        footprint_path = f"{scenario}_footprint.bird"
        print(f"\n--- ENCODING ---")
        try:
            footprint_bytes = encoder.encode(input_data)
        except Exception as e:
            print(f"FAILED: {e}")
            continue

        with open(footprint_path, "wb") as f:
            f.write(footprint_bytes)
            
        print(f"Compressed enormous dataset down to EXACTLY: {len(footprint_bytes)} bytes")
        print(f"Saved to: {footprint_path}")
        
        # 3. Decode & Synthesize to Bird Song
        print(f"\n--- DECODING & SYNTHESIS ---")
        wav_path = f"{scenario}_bird_song.wav"
        
        try:
            with open(footprint_path, "rb") as f:
                read_footprint = f.read()
                
            decoder.decode_to_wav(read_footprint, wav_path)
            
        except Exception as e:
            print(f"FAILED: {e}")
            continue
            
    print("\n" + "=" * 60)
    print("DEMO COMPLETE: Dynamic transforms applied based on Shannon Entropy.")
    print("Check out the generated .wav files representing the data patterns!")
    print("=" * 60)

if __name__ == "__main__":
    run_demo()
