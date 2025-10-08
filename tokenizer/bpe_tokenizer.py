"""
BPE tokenizer for English and code
Byte-level BPE for better handling of technical terms
"""

import os
import sentencepiece as spm
from typing import List, Optional, Dict, Any
import re


class BPETokenizer:
    """
    Byte-level BPE tokenizer for English and code
    
    Advantages:
    - Better handling of technical terms
    - Works well with code and mixed content
    - Byte-level encoding reduces OOV issues
    """
    
    def __init__(
        self,
        model_file: Optional[str] = None,
        vocab_size: int = 16000,
        character_coverage: float = 0.9995
    ):
        self.model_file = model_file
        self.vocab_size = vocab_size
        self.character_coverage = character_coverage
        self.sp = spm.SentencePieceProcessor()
        
        if model_file and os.path.exists(model_file):
            self.sp.load(model_file)
    
    def train(
        self,
        input_file: str,
        output_prefix: str,
        vocab_size: int = 16000,
        character_coverage: float = 0.9995,
        model_type: str = "bpe",
        normalization_rule_name: str = "nmt_nfkc_cf",
        remove_extra_whitespaces: bool = True,
        vocabulary_output_piece_score: bool = True,
        hard_vocab_limit: bool = True,
        use_all_vocab: bool = False,
        byte_fallback: bool = True,
        split_by_unicode_script: bool = True,
        split_by_whitespace: bool = True,
        split_by_number: bool = True,
        treat_whitespace_as_suffix: bool = False,
        allow_whitespace_only_pieces: bool = False,
        unk_id: int = 0,
        bos_id: int = 1,
        eos_id: int = 2,
        pad_id: int = 3,
        unk_piece: str = "<unk>",
        bos_piece: str = "<s>",
        eos_piece: str = "</s>",
        pad_piece: str = "<pad>",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train BPE tokenizer on English/code text
        
        Args:
            input_file: Path to training text file
            output_prefix: Output prefix for model files
            vocab_size: Vocabulary size
            character_coverage: Character coverage ratio
            **kwargs: Additional training parameters
            
        Returns:
            Training results dictionary
        """
        
        # Training arguments
        train_args = [
            f"--input={input_file}",
            f"--model_prefix={output_prefix}",
            f"--vocab_size={vocab_size}",
            f"--character_coverage={character_coverage}",
            f"--model_type={model_type}",
            f"--normalization_rule_name={normalization_rule_name}",
            f"--remove_extra_whitespaces={str(remove_extra_whitespaces).lower()}",
            f"--vocabulary_output_piece_score={str(vocabulary_output_piece_score).lower()}",
            f"--hard_vocab_limit={str(hard_vocab_limit).lower()}",
            f"--use_all_vocab={str(use_all_vocab).lower()}",
            f"--byte_fallback={str(byte_fallback).lower()}",
            f"--split_by_unicode_script={str(split_by_unicode_script).lower()}",
            f"--split_by_whitespace={str(split_by_whitespace).lower()}",
            f"--split_by_number={str(split_by_number).lower()}",
            f"--treat_whitespace_as_suffix={str(treat_whitespace_as_suffix).lower()}",
            f"--allow_whitespace_only_pieces={str(allow_whitespace_only_pieces).lower()}",
            f"--unk_id={unk_id}",
            f"--bos_id={bos_id}",
            f"--eos_id={eos_id}",
            f"--pad_id={pad_id}",
            f"--unk_piece={unk_piece}",
            f"--bos_piece={bos_piece}",
            f"--eos_piece={eos_piece}",
            f"--pad_piece={pad_piece}",
        ]
        
        # Add additional arguments
        for key, value in kwargs.items():
            train_args.append(f"--{key}={value}")
        
        # Train the model
        spm.SentencePieceTrainer.train(" ".join(train_args))
        
        # Load the trained model
        self.sp.load(f"{output_prefix}.model")
        self.model_file = f"{output_prefix}.model"
        
        return {
            "model_file": f"{output_prefix}.model",
            "vocab_file": f"{output_prefix}.vocab",
            "vocab_size": self.sp.get_piece_size(),
            "training_args": train_args
        }
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encode text to token IDs
        
        Args:
            text: Input text
            add_special_tokens: Whether to add special tokens
            
        Returns:
            List of token IDs
        """
        if add_special_tokens:
            return self.sp.encode(text, out_type=int, add_bos=True, add_eos=True)
        else:
            return self.sp.encode(text, out_type=int, add_bos=False, add_eos=False)
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs to text
        
        Args:
            token_ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens
            
        Returns:
            Decoded text
        """
        return self.sp.decode(token_ids, out_type=str)
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text to subword units
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        return self.sp.encode(text, out_type=str)
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size"""
        return self.sp.get_piece_size()
    
    def get_vocab(self) -> Dict[str, int]:
        """Get vocabulary mapping"""
        vocab = {}
        for i in range(self.get_vocab_size()):
            piece = self.sp.id_to_piece(i)
            vocab[piece] = i
        return vocab
    
    def id_to_token(self, token_id: int) -> str:
        """Convert token ID to token string"""
        return self.sp.id_to_piece(token_id)
    
    def token_to_id(self, token: str) -> int:
        """Convert token string to token ID"""
        return self.sp.piece_to_id(token)
    
    def is_english_or_code(self, text: str) -> bool:
        """
        Simple heuristic to detect English/code text
        
        Args:
            text: Input text
            
        Returns:
            True if text appears to be English or code
        """
        # Count Latin characters
        latin_chars = len(re.findall(r'[a-z]', text.lower()))
        total_chars = len(re.findall(r'[а-яёa-z]', text.lower()))
        
        if total_chars == 0:
            return False
        
        # If more than 50% Latin, consider it English/code
        return (latin_chars / total_chars) > 0.5
    
    def is_code_text(self, text: str) -> bool:
        """
        Detect if text contains code patterns
        
        Args:
            text: Input text
            
        Returns:
            True if text appears to contain code
        """
        # Common code patterns
        code_patterns = [
            r'def\s+\w+\s*\(',  # Python function
            r'function\s+\w+\s*\(',  # JavaScript function
            r'class\s+\w+',  # Class definition
            r'import\s+\w+',  # Import statement
            r'from\s+\w+\s+import',  # From import
            r'#include\s*<',  # C++ include
            r'public\s+class',  # Java class
            r'const\s+\w+\s*=',  # JavaScript const
            r'let\s+\w+\s*=',  # JavaScript let
            r'var\s+\w+\s*=',  # JavaScript var
            r'if\s*\(',  # If statement
            r'for\s*\(',  # For loop
            r'while\s*\(',  # While loop
            r'return\s+',  # Return statement
        ]
        
        # Check for code patterns
        for pattern in code_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        
        # Check for high density of special characters (common in code)
        special_chars = len(re.findall(r'[{}();=<>\[\]]', text))
        total_chars = len(text.strip())
        
        if total_chars == 0:
            return False
        
        # If more than 10% special characters, likely code
        return (special_chars / total_chars) > 0.1
