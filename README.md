# Bird Sonification Generative Compression Algorithm

Welcome to the **Bird Sonification Generative Compression Pipeline**! 

This repository contains an experimental and highly abstract data sonification utility. The tool "compresses" arbitrarily massive datasets into a 1-Kilobyte footprint and procedurally generates an artificial "bird song" (nature sounds) representing the dominant structural patterns and rhythms found within the original data.

## Project Overview

It is mathematically impossible to losslessly compress large amounts of random data (like a Petabyte or Terabyte) into a 1 Kilobyte file. Therefore, this project acts as an artistic, highly lossy **Generative Summarizer**.

1. The algorithm analyzes incoming chunks of data.
2. It dynamically calculates the Shannon Entropy of the data to classify its mathematical structure.
3. Based on the entropy, it selects an optimal mathematical transform (**FFT**, **DCT**, or **DWT**) to extract the top 255 dominant floating-point coefficients.
4. It packs these coefficients into a rigid 1 KB `.bird` file.
5. The decoder reads this `.bird` footprint and maps the data to a frequency range between 2,000 Hz and 8,000 Hz to generate a synthetic `.wav` audio file that sounds like ambient bird chirps and trills.

## Features

- **Dynamic Encoding Logic**:
  - **Low Entropy** (Highly Structured Data): Uses Discrete Cosine Transform (DCT) to capture dominant energy, producing pure, rhythmic audio whistles.
  - **Medium Entropy** (Periodic Data): Uses Fast Fourier Transform (FFT) to capture frequency sweeps, producing melodic FM-synthesized chirps.
  - **High Entropy** (Transient/Noisy Data): Uses pywavelet's Discrete Wavelet Transform (DWT) to isolate massive transient spikes, producing sharp clicks and stochastic trills.

## Usage

### Dependencies

```bash
pip install numpy scipy pywavelets
```

### Running the Demo

The included `demo.py` script automatically generates three massive 10 MB datasets (structured, periodic, and transient noise) to showcase the pipeline.

Simply run:
```bash
python demo.py
```

Check your directory for the generated `.bird` footprint files (exactly 1,024 bytes) and listen to the resulting `.wav` generative audio files!

## File Structure

- `bird_codec/`
  - `encoder.py`: Contains the `NatureTransformEncoder` that calculates entropy and extracts dominant patterns.
  - `decoder_synth.py`: Contains the `NatureSynthDecoder` that reads footprints and synthesizes 16-bit PCM `.wav` data.
- `demo.py`: Executable demonstration script.