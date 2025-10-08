#!/usr/bin/env bash
set -euo pipefail

# Скрипт для подготовки модели с чистым корпусом
# Запускать локально или на мощной машине

MODEL_SIZE=${MODEL_SIZE:-2b}  # 2b или 7b
OUTPUT_DIR=${OUTPUT_DIR:-./artifacts}
CONFIG_PATH="configs/model_config_mistral_${MODEL_SIZE}.json"
CORPUS_DIR=${CORPUS_DIR:-./data/raw_corpus}

echo "[+] Preparing RADON Mistral-${MODEL_SIZE} model with clean corpus..."

# 1. Создать директории
mkdir -p $OUTPUT_DIR/{model,tokenizer,configs}
mkdir -p $CORPUS_DIR

# 2. Подготовка чистого корпуса
echo "[+] Preparing clean corpus..."
if [[ ! -f "$CORPUS_DIR/combined_corpus.txt" ]]; then
    echo "⚠️  No corpus found. Please add your texts to $CORPUS_DIR/"
    echo "    Expected structure:"
    echo "    $CORPUS_DIR/"
    echo "    ├── russian/     # Russian texts"
    echo "    ├── english/     # English texts"
    echo "    └── code/        # Code files"
    echo ""
    echo "    Or run: python scripts/prepare_corpus.py --input_dir $CORPUS_DIR --output_dir $CORPUS_DIR"
    exit 1
fi

# 3. Обучить гибридный токенизатор на чистом корпусе
echo "[+] Training hybrid tokenizer on clean corpus..."
python tokenizer/train_hybrid_tokenizer.py \
    --input_file $CORPUS_DIR/combined_corpus.txt \
    --output_dir $OUTPUT_DIR/tokenizer \
    --vocab_size 32000 \
    --russian_ratio 0.4 \
    --english_ratio 0.4 \
    --code_ratio 0.2

# 4. Создать модель с нуля (чистая инициализация)
echo "[+] Initializing clean Mistral-${MODEL_SIZE} model..."
python scripts/initialize_mistral.py \
    --config_path $CONFIG_PATH \
    --output_dir $OUTPUT_DIR/model \
    --model_size $MODEL_SIZE

# 5. Обучение модели на чистом корпусе (опционально)
if [[ "${TRAIN_MODEL:-false}" == "true" ]]; then
    echo "[+] Training model on clean corpus..."
    python scripts/train_mistral.py \
        --config_path $CONFIG_PATH \
        --data_file $CORPUS_DIR/combined_corpus.txt \
        --output_dir $OUTPUT_DIR/model \
        --num_epochs 3 \
        --batch_size 4
fi

# 6. Конвертировать в Safetensors
echo "[+] Converting to Safetensors..."
python scripts/convert_safetensors.py \
    --config_path $CONFIG_PATH \
    --model_path $OUTPUT_DIR/model \
    --output_path $OUTPUT_DIR/model/model.safetensors \
    --model_type mistral

# 7. Копировать конфиги
cp $CONFIG_PATH $OUTPUT_DIR/configs/
cp configs/training_config.json $OUTPUT_DIR/configs/

# 8. Создать архив для загрузки
echo "[+] Creating deployment archive..."
cd $OUTPUT_DIR
tar -czf ../radon-mistral-${MODEL_SIZE}-clean.tar.gz .
cd ..

echo "[+] Model preparation complete!"
echo "[+] Archive: radon-mistral-${MODEL_SIZE}-clean.tar.gz"
echo "[+] Size: $(du -h radon-mistral-${MODEL_SIZE}-clean.tar.gz | cut -f1)"
echo "[+] Model is clean and ready for deployment!"
