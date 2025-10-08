"""
Training script for hybrid Unigram+BPE tokenizer
Combines Russian Unigram and English/code BPE tokenizers
"""

import os
import argparse
from typing import Dict, Any, List
from .unigram_tokenizer import UnigramTokenizer
from .bpe_tokenizer import BPETokenizer
from .hybrid_tokenizer import HybridTokenizer


def prepare_training_data(
    input_file: str,
    output_dir: str,
    russian_ratio: float = 0.4,
    english_ratio: float = 0.4,
    code_ratio: float = 0.2
) -> Dict[str, str]:
    """
    Prepare training data by separating Russian, English, and code text
    
    Args:
        input_file: Path to input text file
        output_dir: Output directory for separated files
        russian_ratio: Ratio of Russian text to include
        english_ratio: Ratio of English text to include
        code_ratio: Ratio of code text to include
        
    Returns:
        Dictionary with paths to separated files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize tokenizers for language detection
    unigram_tokenizer = UnigramTokenizer()
    bpe_tokenizer = BPETokenizer()
    
    # Separate texts
    russian_texts = []
    english_texts = []
    code_texts = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Detect language
            if bpe_tokenizer.is_code_text(line):
                code_texts.append(line)
            elif unigram_tokenizer.is_russian_text(line):
                russian_texts.append(line)
            elif bpe_tokenizer.is_english_or_code(line):
                english_texts.append(line)
            else:
                # Default to English for mixed/unknown
                english_texts.append(line)
    
    # Calculate target sizes
    total_texts = len(russian_texts) + len(english_texts) + len(code_texts)
    target_russian = int(total_texts * russian_ratio)
    target_english = int(total_texts * english_ratio)
    target_code = int(total_texts * code_ratio)
    
    # Sample texts
    import random
    random.shuffle(russian_texts)
    random.shuffle(english_texts)
    random.shuffle(code_texts)
    
    russian_texts = russian_texts[:target_russian]
    english_texts = english_texts[:target_english]
    code_texts = code_texts[:target_code]
    
    # Write separated files
    russian_file = os.path.join(output_dir, "russian.txt")
    english_file = os.path.join(output_dir, "english.txt")
    code_file = os.path.join(output_dir, "code.txt")
    
    with open(russian_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(russian_texts))
    
    with open(english_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(english_texts))
    
    with open(code_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(code_texts))
    
    return {
        "russian": russian_file,
        "english": english_file,
        "code": code_file
    }


def train_hybrid_tokenizer(
    input_file: str,
    output_dir: str,
    vocab_size: int = 32000,
    unigram_vocab_size: int = 16000,
    bpe_vocab_size: int = 16000,
    character_coverage: float = 0.9995,
    **kwargs
) -> Dict[str, Any]:
    """
    Train hybrid tokenizer combining Unigram and BPE
    
    Args:
        input_file: Path to training text file
        output_dir: Output directory for tokenizer files
        vocab_size: Total vocabulary size
        unigram_vocab_size: Unigram vocabulary size
        bpe_vocab_size: BPE vocabulary size
        character_coverage: Character coverage ratio
        **kwargs: Additional training parameters
        
    Returns:
        Training results dictionary
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare training data
    print("Preparing training data...")
    data_files = prepare_training_data(input_file, output_dir)
    
    # Train Unigram tokenizer on Russian text
    print("Training Unigram tokenizer on Russian text...")
    unigram_tokenizer = UnigramTokenizer()
    unigram_results = unigram_tokenizer.train(
        input_file=data_files["russian"],
        output_prefix=os.path.join(output_dir, "unigram"),
        vocab_size=unigram_vocab_size,
        character_coverage=character_coverage,
        **kwargs
    )
    
    # Train BPE tokenizer on English/code text
    print("Training BPE tokenizer on English/code text...")
    bpe_tokenizer = BPETokenizer()
    bpe_results = bpe_tokenizer.train(
        input_file=data_files["english"],
        output_prefix=os.path.join(output_dir, "bpe"),
        vocab_size=bpe_vocab_size,
        character_coverage=character_coverage,
        **kwargs
    )
    
    # Create hybrid tokenizer
    print("Creating hybrid tokenizer...")
    hybrid_tokenizer = HybridTokenizer(
        unigram_model_file=unigram_results["model_file"],
        bpe_model_file=bpe_results["model_file"],
        vocab_size=vocab_size
    )
    
    # Save hybrid tokenizer
    hybrid_tokenizer.save_pretrained(output_dir)
    
    return {
        "unigram_results": unigram_results,
        "bpe_results": bpe_results,
        "hybrid_vocab_size": hybrid_tokenizer.vocab_size,
        "output_dir": output_dir,
        "data_files": data_files
    }


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="Train hybrid Unigram+BPE tokenizer")
    
    # Input/Output
    parser.add_argument("--input", required=True, help="Path to input text file")
    parser.add_argument("--output", required=True, help="Output directory")
    
    # Vocabulary sizes
    parser.add_argument("--vocab_size", type=int, default=32000, help="Total vocabulary size")
    parser.add_argument("--unigram_vocab_size", type=int, default=16000, help="Unigram vocabulary size")
    parser.add_argument("--bpe_vocab_size", type=int, default=16000, help="BPE vocabulary size")
    
    # Training parameters
    parser.add_argument("--character_coverage", type=float, default=0.9995, help="Character coverage")
    parser.add_argument("--normalization_rule_name", default="nmt_nfkc_cf", help="Normalization rule")
    parser.add_argument("--remove_extra_whitespaces", action="store_true", help="Remove extra whitespaces")
    parser.add_argument("--byte_fallback", action="store_true", help="Enable byte fallback")
    
    # Data ratios
    parser.add_argument("--russian_ratio", type=float, default=0.4, help="Ratio of Russian text")
    parser.add_argument("--english_ratio", type=float, default=0.4, help="Ratio of English text")
    parser.add_argument("--code_ratio", type=float, default=0.2, help="Ratio of code text")
    
    args = parser.parse_args()
    
    # Train tokenizer
    results = train_hybrid_tokenizer(
        input_file=args.input,
        output_dir=args.output,
        vocab_size=args.vocab_size,
        unigram_vocab_size=args.unigram_vocab_size,
        bpe_vocab_size=args.bpe_vocab_size,
        character_coverage=args.character_coverage,
        normalization_rule_name=args.normalization_rule_name,
        remove_extra_whitespaces=args.remove_extra_whitespaces,
        byte_fallback=args.byte_fallback
    )
    
    print("Training completed!")
    print(f"Unigram vocabulary size: {results['unigram_results']['vocab_size']}")
    print(f"BPE vocabulary size: {results['bpe_results']['vocab_size']}")
    print(f"Hybrid vocabulary size: {results['hybrid_vocab_size']}")
    print(f"Output directory: {results['output_dir']}")


if __name__ == "__main__":
    main()
