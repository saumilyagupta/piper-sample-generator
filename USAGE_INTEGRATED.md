# Integrated Phonetic Variations - Usage

Wake word training dataset generation now integrated directly into pipeline.

## Quick Start

Generate "hey limbo" training dataset with positive & negative variations:

```bash
python3 -m piper_sample_generator 'hey limbo' \
  --model models/en-us-libritts-high.pt \
  --max-samples 10 \
  --output-dir training_dataset \
  --generate-variations
```

**Output structure:**
```
training_dataset/
├── positive/          # Positive variation samples
├── negative/          # Negative variation samples
```

## Full Example

```bash
# Generate 480 samples (240 positive + 240 negative)
python3 -m piper_sample_generator 'hey limbo' \
  --model models/en-us-libritts-high.pt \
  --max-samples 480 \
  --output-dir wake_word_dataset \
  --batch-size 10 \
  --max-speakers 200 \
  --length-scales 0.75 1.0 1.25 \
  --slerp-weights 0.3 0.5 0.7 \
  --generate-variations
```

## Variations Generated

**Positive (15):** Minor phonetic differences (acceptable positives)
- `hey limbo`, `hey rimbo`, `hey lembo`, `hey limboo`, etc.

**Negative (33):** Significant changes (must not trigger)
- `hey nimbo`, `hey`, `hello limbo`, `limbo`, etc.

## API Usage

```python
from piper_sample_generator import generate_wakeword_variations

phrase = "hey limbo"
positive, negative = generate_wakeword_variations(phrase)

print(f"Positive: {positive}")  # List[str]
print(f"Negative: {negative}")  # List[str]
```

## Workflow

1. **Generate variations + TTS:**
   ```bash
   python3 -m piper_sample_generator 'hey limbo' \
     --model models/en-us-libritts-high.pt \
     --max-samples 100 \
     --output-dir dataset \
     --generate-variations
   ```

2. **Augment with room acoustics:**
   ```bash
   python3 -m piper_sample_generator.augment \
     --sample-rate 16000 \
     dataset/positive dataset/positive_aug/
   
   python3 -m piper_sample_generator.augment \
     --sample-rate 16000 \
     dataset/negative dataset/negative_aug/
   ```

3. **Train wake word detector:**
   ```bash
   openwakeword \
     --positive_dir dataset/positive_aug \
     --negative_dir dataset/negative_aug
   ```

## Parameters

| Flag | Default | Meaning |
|------|---------|---------|
| `--generate-variations` | false | Enable variation generation + separate positive/negative samples |
| `--max-samples` | required | Total samples to generate (split across variations) |
| `--batch-size` | 1 | GPU batch size |
| `--max-speakers` | none | Max LibriTTS speakers (max 904) |
| `--length-scales` | 1.0 0.75 1.25 1.4 | Speaking speeds |
| `--slerp-weights` | 0.5 | Speaker blend weights (.pt only) |
| `--noise-scales` | 0.667 0.75 ... | Audio variation levels |

## Examples

### Quick test
```bash
python3 -m piper_sample_generator 'hello' \
  --model models/en-us-libritts-high.pt \
  --max-samples 20 \
  --generate-variations
```

### Production dataset
```bash
python3 -m piper_sample_generator 'alexa' \
  --model models/en-us-libritts-high.pt \
  --max-samples 1000 \
  --batch-size 20 \
  --max-speakers 500 \
  --output-dir alexa_dataset \
  --generate-variations
```

### Multiple wake words
```bash
for word in "alexa" "google" "hey siri"; do
  python3 -m piper_sample_generator "$word" \
    --model models/en-us-libritts-high.pt \
    --max-samples 100 \
    --output-dir "dataset_$word" \
    --generate-variations
done
```

## Implementation

**Files modified:**
- `piper_sample_generator/__main__.py` - Added `--generate-variations` flag and logic
- `piper_sample_generator/variations.py` - Input validation + type hints
- `piper_sample_generator/__init__.py` - Public API export

**Key features:**
- Automatic phonetic variation generation
- Positive/negative separation
- Integrated into main CLI
- Public Python API via `from piper_sample_generator import generate_wakeword_variations`
