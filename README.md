# Jazz Subgenre Classification (CNNs + Transfer Learning)
A deep learning project that classifies four jazz subgenres from raw audio by converting clips into mel spectrogram images and classifying them with convolutional neural networks. Built for a deep learning course.

## Overview

Most music classifiers treat jazz as a single genre. This project tackles the harder *intra-genre* problem: distinguishing four subgenres that share instrumentation (piano, bass, drums, horns) and differ only in subtle rhythmic, harmonic, and timbral ways:

- **Bebop** — fast tempos, complex chord changes
- **Cool Jazz** — relaxed tempos, smoother timbres
- **Swing** — strong backbeat, big-band arrangements
- **Bossa Nova** — Brazilian samba-jazz fusion, syncopated guitar

The approach converts audio into spectrogram images and applies image-classification CNNs with transfer learning — a standard pipeline in music information retrieval (MIR).

## Dataset

A **self-collected, balanced dataset** of 400 clips:

- 25 songs per subgenre sourced from YouTube and converted to .wav
- 4 non-overlapping 30-second clips extracted per song using **Librosa** (100 clips per class, 400 total)
- All audio resampled to 22,050 Hz
- Each clip is rendered as an image (mel spectrogram, MFCC, or chromagram) for input to the model

## Methods

- **Feature representations compared:** mel spectrogram (128-bin, STFT n_fft=2048, hop=512), 40 MFCCs, and chromagram
- **Architectures compared:** a custom 3-block CNN trained from scratch, a fine-tuned **ResNet-18** (ImageNet-pretrained), and a fine-tuned **ResNet-50**
- **Training:** Adam (lr=0.001, weight_decay=1e-4), cross-entropy loss, batch size 32, ReduceLROnPlateau scheduler, early stopping (patience=5), up to 25 epochs
- **Augmentation:** random horizontal flip and random erasing on training images
- **Validation:** stratified 80/20 split, plus 5-fold stratified cross-validation on the best configuration
- **Stack:** Python, PyTorch, Librosa, scikit-learn, NumPy

## Results

**Architecture comparison (mel spectrogram):**

| Model | Validation Accuracy | Parameters |
|---|---|---|
| Custom CNN (from scratch) | 38.8% | 25.8M |
| ResNet-18 (transfer) | **85.0%** | 11.2M |
| ResNet-50 (transfer) | 82.5% | 23.5M |

**Feature representation comparison (ResNet-18):** mel spectrogram 85.0% > MFCC 78.8% > chromagram 53.8%

**Cross-validation (ResNet-18, mel spectrogram):** 90.8% ± 3.8% accuracy across 5 folds

## Key Takeaways

- **Transfer learning is critical on small datasets.** A from-scratch CNN barely beat the 25% random baseline (38.8%), while a pretrained ResNet-18 reached 85% — the ImageNet features transfer well to spectrograms.
- **Mel spectrograms capture the most useful signal.** Jazz subgenres differ mainly in rhythmic energy and timbre (captured by mel spectrograms) rather than pitch class (captured by chromagrams), which explains the chromagram's weak performance.
- **Bigger isn't better with limited data.** ResNet-50 underperformed the smaller ResNet-18 and took longer to train.

## Limitations

- Only 400 clips, which likely undersample the variation across jazz eras and recording conditions.
- The clip-level train/validation split allows clips from the same song to fall on both sides, risking data leakage; a group-based split (keeping each song's clips together) would be more rigorous.

## Repository Contents

*training notebook*
