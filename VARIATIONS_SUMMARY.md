# Wake Word Variations System - Summary

## What Was Added

Three new tools for generating training datasets with automatic positive/negative phonetic variations:

### 1. **variations.py** - Variation Generator
Generates phonetic variations of any wake word.

```bash
python3 -m piper_sample_generator.variations
```

**Output for "hey limbo":**
- **Positive (15)**: Minor phonetic changes (acceptable false positives)
  - `hey limbo`, `hey rimbo`, `hey lembo`, `hey limboo`, etc.
  
- **Negative (33)**: Significant changes (must not trigger)
  - `hey nimbo`, `hey`, `hello limbo`, `limbo hey`, etc.

### 2. **generate_training_dataset_pt.py** - Full Pipeline
Generates complete training dataset with TTS variations.

```bash
python3 generate_training_dataset_pt.py 'hey limbo' \
  --model models/en-us-libritts-high.pt \
  --samples-per-variation 10 \
  --batch-size 10 \
  --output-dir training_data
```

**Output structure:**
```
training_data/
├── positive/        # 150 positive WAV files
├── negative/        # 330 negative WAV files
```

### 3. **generate_training_dataset.py** - ONNX Version
Same as above but for Piper ONNX voice models (faster inference).

## Workflow

```
┌─────────────────┐
│   Wake Word     │
│  "hey limbo"    │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  Generate Phonetic Variations       │
│  • Positive: minor phonetic changes │
│  • Negative: significant changes    │
└────────┬────────────────────────────┘
         │
         ├─→ 15 positive phrases
         └─→ 33 negative phrases
         │
         ▼
┌─────────────────────────────────────┐
│  Synthesize TTS Audio               │
│  (10 samples per variation)          │
│  • Multiple speakers (speaker mix)   │
│  • Speed variations (length_scale)   │
│  • Noise variation (noise_scale)     │
└────────┬────────────────────────────┘
         │
         ├─→ 150 positive WAV files
         └─→ 330 negative WAV files
         │
         ▼
┌─────────────────────────────────────┐
│  Augment Acoustic Conditions        │
│  python3 -m piper_sample_generator  │
│    .augment --sample-rate 16000 ... │
│  • Volume variation (-12 to 0 dB)   │
│  • Room impulse response            │
│  • Resample to 16kHz                │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  Final Training Dataset             │
│  • 150 positive samples (16kHz)     │
│  • 330 negative samples (16kHz)     │
│  • Total: 480 balanced samples      │
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  Train Wake Word Detector           │
│  (openWakeWord, microWakeWord, etc) │
└─────────────────────────────────────┘
```

## Key Features

### Positive Variations (Minor Phonetic Changes)
Generated phonetic changes that are acceptable as true positives:
- Vowel substitutions: `i→e`, `o→u`, etc.
- Consonant confusion: `l↔r`, `m↔n`, etc.
- Case/spacing variations
- Final vowel additions

**Rationale:** Captures accent differences, speech variation, pronunciation differences.

### Negative Variations (Significant Changes)
Generated changes that should NOT trigger:
- Change initial word: `hey→hello`, `hey→okay`
- Remove words: drop first/last word
- Phonetic confusion: `l→n`, `l→d`
- Character removal/addition
- Word order reversal

**Rationale:** Prevents false positives on confusable phrases and partial matches.

## Parameters for Control

| Parameter | Default | Use |
|-----------|---------|-----|
| `--samples-per-variation` | 5 | Samples per phrase (higher = more diversity) |
| `--batch-size` | 5 | GPU batch processing |
| `--max-speakers` | 100 | LibriTTS speakers to use (max 904) |
| `--length-scales` | 0.75 1.0 1.25 | Speaking speeds |
| `--slerp-weights` | 0.3 0.5 0.7 | Speaker blending (for .pt model) |
| `--noise-scales` | 0.667 | Audio variability |

## Dataset Sizes

Examples for "hey limbo":

| Config | Positive Variations | Samples/Variation | Total Positive | Total Negative | Total |
|--------|-------------------|------------------|-----------------|----------------|-------|
| Quick test | 15 | 2 | 30 | 66 | 96 |
| Small | 15 | 10 | 150 | 330 | 480 |
| Medium | 15 | 50 | 750 | 1,650 | 2,400 |
| Large | 15 | 200 | 3,000 | 6,600 | 9,600 |

## Implementation Details

### Positive Variations Algorithm
1. Start with input phrase
2. Apply vowel substitutions from vowel_variants dict
3. Apply consonant substitutions from consonant_variants dict
4. Add case/spacing variations
5. Add final vowel variations

### Negative Variations Algorithm
1. Keep all positive variations generated
2. Remove individual words
3. Replace first word with common wake words
4. Apply more aggressive phonetic substitutions
5. Remove/insert characters
6. Reverse word order
7. Remove any overlap with positive set

### TTS Diversity
For each variation phrase, generate multiple audio samples by:
- Cycling through different speaker pairs (SLERP blending)
- Cycling through length_scales (speaking speeds)
- Cycling through noise_scales (audio variation)
- Batch processing for efficiency

## Next Steps

1. **Generate dataset:**
   ```bash
   python3 generate_training_dataset_pt.py 'hey limbo' \
     --model models/en-us-libritts-high.pt \
     --samples-per-variation 20 \
     --batch-size 10 \
     --output-dir wake_word_dataset
   ```

2. **Augment with room acoustics:**
   ```bash
   python3 -m piper_sample_generator.augment \
     --sample-rate 16000 \
     wake_word_dataset/positive \
     wake_word_dataset/positive_aug/
   ```

3. **Train detector** (e.g., openWakeWord):
   ```bash
   openwakeword \
     --positive_dir wake_word_dataset/positive_aug \
     --negative_dir wake_word_dataset/negative_aug
   ```

## Files Added

- `piper_sample_generator/variations.py` - Phonetic variation generator
- `generate_training_dataset_pt.py` - PyTorch model training data generator
- `generate_training_dataset.py` - ONNX model training data generator
- `TRAINING_DATASET.md` - Complete usage guide
- `VARIATIONS_SUMMARY.md` - This file

## Advantages

✓ Automated generation of confusable negative samples
✓ Multiple TTS speaker variations for robustness
✓ Phonetic-based classification (positive vs negative)
✓ Scalable to any wake word phrase
✓ Includes acoustic augmentation pipeline
✓ Balanced dataset generation (configurable ratio)
✓ Ready for immediate model training
