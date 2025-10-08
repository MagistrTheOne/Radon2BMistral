"""
Подготовка чистого корпуса для обучения RADON
"""

import os
import re
import json
from pathlib import Path
from typing import List, Dict, Any
import argparse


def clean_text(text: str) -> str:
    """Очистка текста от мусора"""
    
    # Удаление HTML тегов
    text = re.sub(r'<[^>]+>', '', text)
    
    # Удаление URL
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    # Удаление email
    text = re.sub(r'\S+@\S+', '', text)
    
    # Удаление лишних пробелов и переносов
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n+', '\n', text)
    
    # Удаление специальных символов (оставляем только буквы, цифры, пунктуацию)
    text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\"\']', '', text)
    
    return text.strip()


def prepare_russian_corpus(
    input_dir: str,
    output_file: str,
    min_length: int = 50,
    max_length: int = 2000
) -> Dict[str, Any]:
    """Подготовка русского корпуса"""
    
    print("[+] Preparing Russian corpus...")
    
    russian_texts = []
    total_chars = 0
    
    # Поиск текстовых файлов
    for file_path in Path(input_dir).rglob("*.txt"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Очистка текста
            cleaned = clean_text(content)
            
            # Фильтрация по длине
            if min_length <= len(cleaned) <= max_length:
                russian_texts.append(cleaned)
                total_chars += len(cleaned)
                
        except Exception as e:
            print(f"⚠️  Error processing {file_path}: {e}")
            continue
    
    # Сохранение корпуса
    with open(output_file, 'w', encoding='utf-8') as f:
        for text in russian_texts:
            f.write(text + '\n\n')
    
    stats = {
        "total_texts": len(russian_texts),
        "total_chars": total_chars,
        "avg_length": total_chars // len(russian_texts) if russian_texts else 0
    }
    
    print(f"✅ Russian corpus: {stats['total_texts']} texts, {stats['total_chars']:,} chars")
    return stats


def prepare_english_corpus(
    input_dir: str,
    output_file: str,
    min_length: int = 50,
    max_length: int = 2000
) -> Dict[str, Any]:
    """Подготовка английского корпуса"""
    
    print("[+] Preparing English corpus...")
    
    english_texts = []
    total_chars = 0
    
    # Поиск текстовых файлов
    for file_path in Path(input_dir).rglob("*.txt"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Очистка текста
            cleaned = clean_text(content)
            
            # Фильтрация по длине
            if min_length <= len(cleaned) <= max_length:
                english_texts.append(cleaned)
                total_chars += len(cleaned)
                
        except Exception as e:
            print(f"⚠️  Error processing {file_path}: {e}")
            continue
    
    # Сохранение корпуса
    with open(output_file, 'w', encoding='utf-8') as f:
        for text in english_texts:
            f.write(text + '\n\n')
    
    stats = {
        "total_texts": len(english_texts),
        "total_chars": total_chars,
        "avg_length": total_chars // len(english_texts) if english_texts else 0
    }
    
    print(f"✅ English corpus: {stats['total_texts']} texts, {stats['total_chars']:,} chars")
    return stats


def prepare_code_corpus(
    input_dir: str,
    output_file: str,
    min_length: int = 20,
    max_length: int = 1000
) -> Dict[str, Any]:
    """Подготовка корпуса кода"""
    
    print("[+] Preparing code corpus...")
    
    code_extensions = ['.py', '.js', '.ts', '.java', '.cpp', '.c', '.h', '.go', '.rs']
    code_texts = []
    total_chars = 0
    
    # Поиск файлов кода
    for ext in code_extensions:
        for file_path in Path(input_dir).rglob(f"*{ext}"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Очистка кода (удаление комментариев, лишних пробелов)
                cleaned = clean_text(content)
                
                # Фильтрация по длине
                if min_length <= len(cleaned) <= max_length:
                    code_texts.append(cleaned)
                    total_chars += len(cleaned)
                    
            except Exception as e:
                print(f"⚠️  Error processing {file_path}: {e}")
                continue
    
    # Сохранение корпуса
    with open(output_file, 'w', encoding='utf-8') as f:
        for text in code_texts:
            f.write(text + '\n\n')
    
    stats = {
        "total_texts": len(code_texts),
        "total_chars": total_chars,
        "avg_length": total_chars // len(code_texts) if code_texts else 0
    }
    
    print(f"✅ Code corpus: {stats['total_texts']} texts, {stats['total_chars']:,} chars")
    return stats


def create_combined_corpus(
    russian_file: str,
    english_file: str,
    code_file: str,
    output_file: str,
    russian_ratio: float = 0.4,
    english_ratio: float = 0.4,
    code_ratio: float = 0.2
) -> Dict[str, Any]:
    """Создание объединенного корпуса"""
    
    print("[+] Creating combined corpus...")
    
    # Чтение корпусов
    with open(russian_file, 'r', encoding='utf-8') as f:
        russian_texts = [line.strip() for line in f if line.strip()]
    
    with open(english_file, 'r', encoding='utf-8') as f:
        english_texts = [line.strip() for line in f if line.strip()]
    
    with open(code_file, 'r', encoding='utf-8') as f:
        code_texts = [line.strip() for line in f if line.strip()]
    
    # Вычисление количества текстов для каждого языка
    total_texts = len(russian_texts) + len(english_texts) + len(code_texts)
    russian_count = int(total_texts * russian_ratio)
    english_count = int(total_texts * english_ratio)
    code_count = int(total_texts * code_ratio)
    
    # Создание объединенного корпуса
    combined_texts = []
    
    # Добавление русских текстов
    combined_texts.extend(russian_texts[:russian_count])
    
    # Добавление английских текстов
    combined_texts.extend(english_texts[:english_count])
    
    # Добавление кода
    combined_texts.extend(code_texts[:code_count])
    
    # Перемешивание
    import random
    random.shuffle(combined_texts)
    
    # Сохранение
    with open(output_file, 'w', encoding='utf-8') as f:
        for text in combined_texts:
            f.write(text + '\n\n')
    
    stats = {
        "total_texts": len(combined_texts),
        "russian_texts": russian_count,
        "english_texts": english_count,
        "code_texts": code_count,
        "total_chars": sum(len(text) for text in combined_texts)
    }
    
    print(f"✅ Combined corpus: {stats['total_texts']} texts, {stats['total_chars']:,} chars")
    return stats


def main():
    parser = argparse.ArgumentParser(description="Prepare clean corpus for RADON")
    parser.add_argument("--input_dir", required=True, help="Input directory with texts")
    parser.add_argument("--output_dir", default="./data", help="Output directory")
    parser.add_argument("--russian_ratio", type=float, default=0.4, help="Russian text ratio")
    parser.add_argument("--english_ratio", type=float, default=0.4, help="English text ratio")
    parser.add_argument("--code_ratio", type=float, default=0.2, help="Code text ratio")
    
    args = parser.parse_args()
    
    # Создание выходной директории
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Подготовка корпусов
    russian_stats = prepare_russian_corpus(
        args.input_dir,
        os.path.join(args.output_dir, "russian_corpus.txt")
    )
    
    english_stats = prepare_english_corpus(
        args.input_dir,
        os.path.join(args.output_dir, "english_corpus.txt")
    )
    
    code_stats = prepare_code_corpus(
        args.input_dir,
        os.path.join(args.output_dir, "code_corpus.txt")
    )
    
    # Создание объединенного корпуса
    combined_stats = create_combined_corpus(
        os.path.join(args.output_dir, "russian_corpus.txt"),
        os.path.join(args.output_dir, "english_corpus.txt"),
        os.path.join(args.output_dir, "code_corpus.txt"),
        os.path.join(args.output_dir, "combined_corpus.txt"),
        args.russian_ratio,
        args.english_ratio,
        args.code_ratio
    )
    
    # Сохранение статистики
    stats = {
        "russian": russian_stats,
        "english": english_stats,
        "code": code_stats,
        "combined": combined_stats
    }
    
    with open(os.path.join(args.output_dir, "corpus_stats.json"), 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    print("\n🎉 Corpus preparation complete!")
    print(f"📊 Statistics saved to {args.output_dir}/corpus_stats.json")


if __name__ == "__main__":
    main()
