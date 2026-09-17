#!/bin/bash
# Full pipeline: generate → augment → train

PHRASE="hey limbo"
DATASET_DIR="hey_limbo_dataset"
MODEL_FILE="models/en-us-libritts-high.pt"
OUTPUT_MODEL="wakeword_model_hey_limbo.pkl"

echo "=========================================="
echo "STEP 1: Generate dataset with variations"
echo "=========================================="
python3 -m piper_sample_generator "$PHRASE" \
  --model "$MODEL_FILE" \
  --max-samples 500 \
  --output-dir "$DATASET_DIR" \
  --batch-size 10 \
  --max-speakers 200 \
  --length-scales 0.75 1.0 1.25 \
  --slerp-weights 0.3 0.5 0.7 \
  --generate-variations

if [ ! -d "$DATASET_DIR/positive" ]; then
  echo "ERROR: Failed to generate positive samples"
  exit 1
fi

echo ""
echo "=========================================="
echo "STEP 2: Augment positive samples"
echo "=========================================="
python3 -m piper_sample_generator.augment \
  --sample-rate 16000 \
  "$DATASET_DIR/positive" \
  "$DATASET_DIR/positive_aug/"

echo ""
echo "=========================================="
echo "STEP 3: Augment negative samples"
echo "=========================================="
python3 -m piper_sample_generator.augment \
  --sample-rate 16000 \
  "$DATASET_DIR/negative" \
  "$DATASET_DIR/negative_aug/"

echo ""
echo "=========================================="
echo "STEP 4: Train wake word detector"
echo "=========================================="
python3 train_wakeword.py \
  "$DATASET_DIR/positive_aug" \
  "$DATASET_DIR/negative_aug" \
  --output "$OUTPUT_MODEL"

echo ""
echo "=========================================="
echo "PIPELINE COMPLETE"
echo "=========================================="
echo "✓ Dataset: $DATASET_DIR/"
echo "✓ Model: $OUTPUT_MODEL"
echo ""
echo "Next: Use model for inference:"
echo "  python3 test_wakeword.py $OUTPUT_MODEL <audio.wav>"
