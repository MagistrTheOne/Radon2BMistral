"""
Training script for SentencePiece tokenizer
"""

import os
import sentencepiece as spm
from typing import List, Optional, Dict, Any
import json


def train_sentencepiece_tokenizer(
    input_file: str,
    output_dir: str,
    vocab_size: int = 32000,
    model_prefix: str = "tokenizer",
    character_coverage: float = 0.9995,
    model_type: str = "bpe",
    split_by_unicode_script: bool = True,
    split_by_whitespace: bool = True,
    split_by_number: bool = True,
    treat_whitespace_as_suffix: bool = False,
    allow_whitespace_only_pieces: bool = False,
    normalization_rule_name: str = "nmt_nfkc_cf",
    remove_extra_whitespaces: bool = True,
    vocabulary_output_piece_score: bool = True,
    hard_vocab_limit: bool = True,
    use_all_vocab: bool = False,
    byte_fallback: bool = True,
    required_chars: str = "",
    unk_id: int = 0,
    bos_id: int = 1,
    eos_id: int = 2,
    pad_id: int = 3,
    unk_piece: str = "<unk>",
    bos_piece: str = "<s>",
    eos_piece: str = "</s>",
    pad_piece: str = "<pad>",
    unk_surface: str = " \342\201\205 ",
    train_extremely_large_corpus: bool = False,
    seed_sentencepiece_size: int = 1000000,
    shrinking_factor: float = 0.75,
    num_threads: int = 16,
    num_sub_iterations: int = 2,
    max_sentencepiece_length: int = 16,
    max_sentence_length: int = 4192,
    shuffle_input_sentence: bool = True,
    input_sentence_size: int = 0,
    max_sentencepiece_length: int = 16,
    split_digits: bool = False,
    pretokenization_delimiter: str = "",
    seed_sentencepiece_size: int = 1000000,
    shrinking_factor: float = 0.75,
    num_threads: int = 16,
    num_sub_iterations: int = 2,
    max_sentencepiece_length: int = 16,
    max_sentence_length: int = 4192,
    shuffle_input_sentence: bool = True,
    input_sentence_size: int = 0,
    max_sentencepiece_length: int = 16,
    split_digits: bool = False,
    pretokenization_delimiter: str = "",
    **kwargs
) -> Dict[str, Any]:
    """
    Train a SentencePiece tokenizer
    
    Args:
        input_file: Path to input text file
        output_dir: Directory to save tokenizer files
        vocab_size: Size of vocabulary
        model_prefix: Prefix for model files
        **kwargs: Additional SentencePiece parameters
    
    Returns:
        Dictionary with training results and file paths
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare training arguments
    train_args = {
        'input': input_file,
        'model_prefix': os.path.join(output_dir, model_prefix),
        'vocab_size': vocab_size,
        'character_coverage': character_coverage,
        'model_type': model_type,
        'split_by_unicode_script': split_by_unicode_script,
        'split_by_whitespace': split_by_whitespace,
        'split_by_number': split_by_number,
        'treat_whitespace_as_suffix': treat_whitespace_as_suffix,
        'allow_whitespace_only_pieces': allow_whitespace_only_pieces,
        'normalization_rule_name': normalization_rule_name,
        'remove_extra_whitespaces': remove_extra_whitespaces,
        'vocabulary_output_piece_score': vocabulary_output_piece_score,
        'hard_vocab_limit': hard_vocab_limit,
        'use_all_vocab': use_all_vocab,
        'byte_fallback': byte_fallback,
        'required_chars': required_chars,
        'unk_id': unk_id,
        'bos_id': bos_id,
        'eos_id': eos_id,
        'pad_id': pad_id,
        'unk_piece': unk_piece,
        'bos_piece': bos_piece,
        'eos_piece': eos_piece,
        'pad_piece': pad_piece,
        'unk_surface': unk_surface,
        'train_extremely_large_corpus': train_extremely_large_corpus,
        'seed_sentencepiece_size': seed_sentencepiece_size,
        'shrinking_factor': shrinking_factor,
        'num_threads': num_threads,
        'num_sub_iterations': num_sub_iterations,
        'max_sentencepiece_length': max_sentencepiece_length,
        'max_sentence_length': max_sentence_length,
        'shuffle_input_sentence': shuffle_input_sentence,
        'input_sentence_size': input_sentence_size,
        'split_digits': split_digits,
        'pretokenization_delimiter': pretokenization_delimiter,
    }
    
    # Add any additional kwargs
    train_args.update(kwargs)
    
    # Train the tokenizer
    print(f"Training SentencePiece tokenizer with vocab_size={vocab_size}")
    print(f"Input file: {input_file}")
    print(f"Output directory: {output_dir}")
    
    try:
        spm.SentencePieceTrainer.train(**train_args)
        print("Tokenizer training completed successfully!")
    except Exception as e:
        print(f"Error during tokenizer training: {e}")
        raise
    
    # Prepare results
    model_path = os.path.join(output_dir, f"{model_prefix}.model")
    vocab_path = os.path.join(output_dir, f"{model_prefix}.vocab")
    
    # Create tokenizer configuration
    tokenizer_config = {
        "model_file": model_path,
        "vocab_file": vocab_path,
        "vocab_size": vocab_size,
        "model_type": model_type,
        "special_tokens": {
            "unk_token": unk_piece,
            "bos_token": bos_piece,
            "eos_token": eos_piece,
            "pad_token": pad_piece,
        },
        "token_ids": {
            "unk_id": unk_id,
            "bos_id": bos_id,
            "eos_id": eos_id,
            "pad_id": pad_id,
        }
    }
    
    # Save tokenizer configuration
    config_path = os.path.join(output_dir, "tokenizer_config.json")
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(tokenizer_config, f, indent=2, ensure_ascii=False)
    
    # Test the trained tokenizer
    sp = spm.SentencePieceProcessor()
    sp.load(model_path)
    
    # Test encoding/decoding
    test_text = "Привет! Hello! This is a test."
    encoded = sp.encode(test_text, out_type=str)
    decoded = sp.decode(encoded)
    
    print(f"Test encoding/decoding:")
    print(f"Original: {test_text}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")
    
    results = {
        "model_path": model_path,
        "vocab_path": vocab_path,
        "config_path": config_path,
        "vocab_size": vocab_size,
        "test_encoding": encoded,
        "test_decoding": decoded,
        "success": True
    }
    
    return results


def train_tokenizer_from_corpus(
    corpus_file: str,
    output_dir: str = "./tokenizer_output",
    vocab_size: int = 32000,
    model_prefix: str = "custom_tokenizer"
) -> Dict[str, Any]:
    """
    Train tokenizer from a corpus file with default settings for multilingual text
    
    Args:
        corpus_file: Path to corpus text file
        output_dir: Output directory for tokenizer files
        vocab_size: Vocabulary size
        model_prefix: Prefix for model files
    
    Returns:
        Training results dictionary
    """
    
    # Default settings optimized for Russian/English text
    return train_sentencepiece_tokenizer(
        input_file=corpus_file,
        output_dir=output_dir,
        vocab_size=vocab_size,
        model_prefix=model_prefix,
        character_coverage=0.9995,
        model_type="bpe",
        split_by_unicode_script=True,
        split_by_whitespace=True,
        split_by_number=True,
        treat_whitespace_as_suffix=False,
        allow_whitespace_only_pieces=False,
        normalization_rule_name="nmt_nfkc_cf",
        remove_extra_whitespaces=True,
        vocabulary_output_piece_score=True,
        hard_vocab_limit=True,
        use_all_vocab=False,
        byte_fallback=True,
        required_chars="",
        unk_id=0,
        bos_id=1,
        eos_id=2,
        pad_id=3,
        unk_piece="<unk>",
        bos_piece="<s>",
        eos_piece="</s>",
        pad_piece="<pad>",
        unk_surface=" \342\201\205 ",
        train_extremely_large_corpus=False,
        seed_sentencepiece_size=1000000,
        shrinking_factor=0.75,
        num_threads=16,
        num_sub_iterations=2,
        max_sentencepiece_length=16,
        max_sentence_length=4192,
        shuffle_input_sentence=True,
        input_sentence_size=0,
        split_digits=False,
        pretokenization_delimiter="",
    )


if __name__ == "__main__":
    """Example usage of tokenizer training"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train SentencePiece tokenizer")
    parser.add_argument("--input", required=True, help="Input corpus file")
    parser.add_argument("--output", default="./tokenizer_output", help="Output directory")
    parser.add_argument("--vocab_size", type=int, default=32000, help="Vocabulary size")
    parser.add_argument("--prefix", default="custom_tokenizer", help="Model prefix")
    
    args = parser.parse_args()
    
    # Train tokenizer
    results = train_tokenizer_from_corpus(
        corpus_file=args.input,
        output_dir=args.output,
        vocab_size=args.vocab_size,
        model_prefix=args.prefix
    )
    
    print(f"\nTokenizer training completed!")
    print(f"Model file: {results['model_path']}")
    print(f"Vocab file: {results['vocab_path']}")
    print(f"Config file: {results['config_path']}")
    print(f"Vocabulary size: {results['vocab_size']}")
