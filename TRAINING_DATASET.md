# Training Dataset Generation with Phonetic Variations

This guide shows how to generate a complete wake word training dataset with automatic positive and negative phonetic variations.

## Overview

**Problem:** Training a wake word detector requires:
- Positive samples (target phrase with variations)
- Negative samples (confusable/different phrases)
- Multiple acoustic conditions
- Sufficient data volume

**Solution:** Automatically generate phonetic variations of any wake word and synthesize audio for all variations.

## Quick Start

### Generate "hey limbo" training dataset

```bash
python3 generate_training_dataset_pt.py 'hey limbo' \
  --model models/en-us-libritts-high.pt \
  --samples-per-variation 10 \
  --batch-size 10 \
  --output-dir training_data
```

This generates:
- **Positive samples**: "hey limbo", "hey rimbo", "hey lembo", etc. (15 variations × 10 = 150 samples)
- **Negative samples**: "hey nimbo", "hey", "limbo", etc. (33 variations × 10 = 330 samples)
- **Total**: 480 audio files (70% negative, 30% positive)

### Add acoustic augmentation

```bash
python3 -m piper_sample_generator.augment \
  --sample-rate 16000 \
  training_data/positive training_data/positive_aug/

python3 -m piper_sample_generator.augment \
  --sample-rate 16000 \
  training_data/negative training_data/negative_aug/
```

This adds:
- Random volume changes (-12 to 0 dB)
- Impulse responses (room acoustics)
- Resampling to 16kHz

## Variation Types

### Positive Variations (Minor phonetic changes)
Close to original phrase - false positives are acceptable

Examples for "hey limbo":
- `hey limbo` (original)
- `hey rimbo` (l→r consonant)
- `hey lembo` (i→e vowel)
- `hey limboo` (extra vowel)
- `heylimbo` (remove space)
- `HEY LIMBO` (case variation)

**Use case:** Capture accents, speech variation, pronunciation differences

### Negative Variations (Significant changes)
Different from target phrase - false negatives must be prevented

Examples for "hey limbo":
- `hey nimbo` (l→n consonant)
- `hey` (missing last word)
- `limbo` (missing first word)
- `hello limbo` (different greeting)
- `limbo hey` (word order)
- `hey lambo` (phonetic confusion)

**Use case:** Prevent triggering on confusable phrases

## Full Workflow

```bash
# 1. Generate variations (positive + negative)
python3 generate_training_dataset_pt.py 'hey limbo' \
  --model models/en-us-libritts-high.pt \
  --samples-per-variation 20 \
  --batch-size 10 \
  --max-speakers 200 \
  --length-scales 0.7 0.85 1.0 1.15 1.3 \
  --slerp-weights 0.25 0.5 0.75 \
  --output-dir wake_word_dataset

# 2. Augment both positive and negative
python3 -m piper_sample_generator.augment \
  --sample-rate 16000 \
  wake_word_dataset/positive \
  wake_word_dataset/positive_aug/

python3 -m piper_sample_generator.augment \
  --sample-rate 16000 \
  wake_word_dataset/negative \
  wake_word_dataset/negative_aug/

# 3. Verify dataset statistics
find wake_word_dataset/positive_aug -name "*.wav" | wc -l  # Count positive
find wake_word_dataset/negative_aug -name "*.wav" | wc -l  # Count negative
```

## Parameters

### Phonetic Variation Control
```bash
# Already handled by variations.py
# Can't customize - but covers most confusable phonetics
```

### TTS Variation Control

**`--samples-per-variation`** (default: 5)
- How many audio files per text phrase variation
- Higher = more speaker/speed/noise diversity
- Example: 15 positive variations × 10 samples = 150 files

**`--batch-size`** (default: 5)
- GPU batch processing (faster on NVIDIA)
- Increase if you have GPU memory

**`--max-speakers`** (default: 100)
- Max LibriTTS speakers to use
- 904 total available; using fewer = faster
- Higher = more speaker variety

**`--length-scales`** (default: 0.75 1.0 1.25)
- Speaking speed: <1 faster, >1 slower
- More values = more diversity

**`--slerp-weights`** (default: 0.3 0.5 0.7)
- Speaker blending: 0=only speaker1, 1=only speaker2
- More weights = smoother speaker transitions

**`--noise-scales`** (default: 0.667)
- Audio variability/graininess
- Recommended: 0.667 (most natural)

## Output Structure

```
training_data/
├── positive/              # 150 WAV files (positive variations)
│   ├── 0.wav
│   ├── 1.wav
│   └── ...
├── positive_aug/          # 150 augmented WAV files
│   ├── 0.wav
│   ├── 1.wav
│   └── ...
├── negative/              # 330 WAV files (negative variations)
│   ├── 0.wav
│   ├── 1.wav
│   └── ...
└── negative_aug/          # 330 augmented WAV files
    ├── 0.wav
    ├── 1.wav
    └── ...
```

## Dataset Sizes

For typical wake word detector training:

| Use case | Positive | Negative | Total |
|----------|----------|----------|-------|
| Quick test | 150 | 150 | 300 |
| Small model | 500 | 1,500 | 2,000 |
| Medium model | 2,000 | 6,000 | 8,000 |
| Large model | 10,000 | 30,000 | 40,000 |

Generate using:
```bash
# 500 positive, 1500 negative = 2000 total
python3 generate_training_dataset_pt.py 'hey limbo' \
  --model models/en-us-libritts-high.pt \
  --samples-per-variation 33 \
  --batch-size 10 \
  --max-speakers 200 \
  --output-dir dataset_2k
```

## Next: Train Wake Word Detector

Use the generated dataset to train:
- [openWakeWord](https://github.com/dscripka/openWakeWord)
- [microWakeWord](https://github.com/kahrendt/microWakeWord)
- Custom CNN/RNN classifier

Example with openWakeWord:
```bash
pip install openwakeword
openwakeword --model_path models/hey_limbo.onnx \
  --positive_dir training_data/positive_aug \
  --negative_dir training_data/negative_aug
```

## Customization

### Add custom negative phrases

```python
from piper_sample_generator.variations import generate_wakeword_variations
from piper_sample_generator import generate_samples

phrase = "hey limbo"
positive, negative = generate_wakeword_variations(phrase)

# Add custom negative phrases
custom_negatives = [
    "okay google",
    "alexa",
    "hey siri",
    "computer",
]
negative.extend(custom_negatives)

# Generate TTS for all
generate_samples(negative, "custom_negatives/", model_path, max_samples=len(negative)*5)
```

### Use Piper ONNX models instead

For faster inference (CPU-friendly):
```bash
python3 generate_training_dataset.py 'hey limbo' \
  --model voices/en_US-lessac-medium.onnx \
  --samples-per-variation 20 \
  --output-dir dataset_piper
```

(Requires downloading Piper voices first - see README.md)
