"""
Hybrid tokenizer combining Unigram (Russian) and BPE (English/Code)
Intelligent routing based on language detection
"""

import os
import json
from typing import List, Optional, Dict, Any, Union
from transformers import PreTrainedTokenizer
from transformers.tokenization_utils_base import BatchEncoding

from .unigram_tokenizer import UnigramTokenizer
from .bpe_tokenizer import BPETokenizer


class HybridTokenizer(PreTrainedTokenizer):
    """
    Hybrid tokenizer for Russian-English ML corpus
    
    Features:
    - Unigram for Russian (better morphology)
    - BPE for English/code (better technical terms)
    - Language detection and routing
    - Domain-specific tokens
    - Combined vocabulary (32K total)
    """
    
    def __init__(
        self,
        unigram_model_file: Optional[str] = None,
        bpe_model_file: Optional[str] = None,
        vocab_size: int = 32000,
        unk_token: str = "<unk>",
        bos_token: str = "<s>",
        eos_token: str = "</s>",
        pad_token: str = "<pad>",
        cls_token: Optional[str] = None,
        sep_token: Optional[str] = None,
        mask_token: Optional[str] = None,
        additional_special_tokens: Optional[List[str]] = None,
        **kwargs
    ):
        # Initialize tokenizers
        self.unigram_tokenizer = UnigramTokenizer(unigram_model_file)
        self.bpe_tokenizer = BPETokenizer(bpe_model_file)
        
        # Special tokens
        self.unk_token = unk_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.cls_token = cls_token
        self.sep_token = sep_token
        self.mask_token = mask_token
        
        # Additional special tokens
        if additional_special_tokens is None:
            additional_special_tokens = [
                "<ML>", "<CODE>", "<MATH>",  # Domain-specific
                "<RU>", "<EN>",  # Language markers
                "<TASK>", "<RESPONSE>",  # Task-specific
            ]
        self.additional_special_tokens = additional_special_tokens
        
        # Combined vocabulary
        self._build_combined_vocab()
        
        # Initialize parent class
        super().__init__(
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            cls_token=cls_token,
            sep_token=sep_token,
            mask_token=mask_token,
            additional_special_tokens=additional_special_tokens,
            **kwargs
        )
    
    def _build_combined_vocab(self):
        """Build combined vocabulary from both tokenizers"""
        self.vocab = {}
        self.ids_to_tokens = {}
        
        # Add special tokens first
        special_tokens = [
            self.unk_token, self.bos_token, self.eos_token, self.pad_token
        ]
        if self.cls_token:
            special_tokens.append(self.cls_token)
        if self.sep_token:
            special_tokens.append(self.sep_token)
        if self.mask_token:
            special_tokens.append(self.mask_token)
        
        special_tokens.extend(self.additional_special_tokens)
        
        # Add special tokens to vocabulary
        for i, token in enumerate(special_tokens):
            self.vocab[token] = i
            self.ids_to_tokens[i] = token
        
        # Add tokens from both tokenizers
        current_id = len(special_tokens)
        
        # Unigram tokens (Russian)
        unigram_vocab = self.unigram_tokenizer.get_vocab()
        for token, _ in unigram_vocab.items():
            if token not in self.vocab:
                self.vocab[token] = current_id
                self.ids_to_tokens[current_id] = token
                current_id += 1
        
        # BPE tokens (English/code)
        bpe_vocab = self.bpe_tokenizer.get_vocab()
        for token, _ in bpe_vocab.items():
            if token not in self.vocab:
                self.vocab[token] = current_id
                self.ids_to_tokens[current_id] = token
                current_id += 1
        
        self.vocab_size = len(self.vocab)
    
    def _detect_language(self, text: str) -> str:
        """
        Detect language and routing strategy
        
        Args:
            text: Input text
            
        Returns:
            Language type: 'russian', 'english', 'code', 'mixed'
        """
        # Check for code patterns first
        if self.bpe_tokenizer.is_code_text(text):
            return 'code'
        
        # Check for Russian
        if self.unigram_tokenizer.is_russian_text(text):
            return 'russian'
        
        # Check for English
        if self.bpe_tokenizer.is_english_or_code(text):
            return 'english'
        
        # Default to mixed
        return 'mixed'
    
    def _route_tokenization(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Route text to appropriate tokenizer
        
        Args:
            text: Input text
            add_special_tokens: Whether to add special tokens
            
        Returns:
            List of token IDs
        """
        language = self._detect_language(text)
        
        if language == 'russian':
            # Use Unigram for Russian
            token_ids = self.unigram_tokenizer.encode(text, add_special_tokens=False)
            # Map to combined vocabulary
            mapped_ids = []
            for token_id in token_ids:
                token = self.unigram_tokenizer.id_to_token(token_id)
                if token in self.vocab:
                    mapped_ids.append(self.vocab[token])
                else:
                    mapped_ids.append(self.vocab[self.unk_token])
            return mapped_ids
        
        elif language in ['english', 'code']:
            # Use BPE for English/code
            token_ids = self.bpe_tokenizer.encode(text, add_special_tokens=False)
            # Map to combined vocabulary
            mapped_ids = []
            for token_id in token_ids:
                token = self.bpe_tokenizer.id_to_token(token_id)
                if token in self.vocab:
                    mapped_ids.append(self.vocab[token])
                else:
                    mapped_ids.append(self.vocab[self.unk_token])
            return mapped_ids
        
        else:  # mixed
            # For mixed text, use BPE as fallback
            token_ids = self.bpe_tokenizer.encode(text, add_special_tokens=False)
            mapped_ids = []
            for token_id in token_ids:
                token = self.bpe_tokenizer.id_to_token(token_id)
                if token in self.vocab:
                    mapped_ids.append(self.vocab[token])
                else:
                    mapped_ids.append(self.vocab[self.unk_token])
            return mapped_ids
    
    def _tokenize(self, text: str, **kwargs) -> List[str]:
        """Tokenize text into tokens"""
        language = self._detect_language(text)
        
        if language == 'russian':
            tokens = self.unigram_tokenizer.tokenize(text)
        else:
            tokens = self.bpe_tokenizer.tokenize(text)
        
        return tokens
    
    def _convert_token_to_id(self, token: str) -> int:
        """Convert token to ID"""
        return self.vocab.get(token, self.vocab[self.unk_token])
    
    def _convert_id_to_token(self, index: int) -> str:
        """Convert ID to token"""
        return self.ids_to_tokens.get(index, self.unk_token)
    
    def convert_tokens_to_ids(self, tokens: Union[str, List[str]]) -> Union[int, List[int]]:
        """Convert tokens to IDs"""
        if isinstance(tokens, str):
            return self._convert_token_to_id(tokens)
        return [self._convert_token_to_id(token) for token in tokens]
    
    def convert_ids_to_tokens(self, ids: Union[int, List[int]], skip_special_tokens: bool = False) -> Union[str, List[str]]:
        """Convert IDs to tokens"""
        if isinstance(ids, int):
            return self._convert_id_to_token(ids)
        return [self._convert_id_to_token(id) for id in ids]
    
    def encode(
        self,
        text: Union[str, List[str]],
        add_special_tokens: bool = True,
        padding: Union[bool, str] = False,
        truncation: Union[bool, str] = False,
        max_length: Optional[int] = None,
        stride: int = 0,
        return_tensors: Optional[str] = None,
        **kwargs
    ) -> Union[List[int], List[List[int]], BatchEncoding]:
        """Encode text to token IDs"""
        
        if isinstance(text, str):
            # Single text
            token_ids = self._route_tokenization(text, add_special_tokens)
            
            # Add special tokens if requested
            if add_special_tokens:
                if self.bos_token:
                    token_ids = [self.vocab[self.bos_token]] + token_ids
                if self.eos_token:
                    token_ids = token_ids + [self.vocab[self.eos_token]]
            
            # Apply padding/truncation
            if padding or truncation:
                token_ids = self._apply_padding_truncation(token_ids, max_length, padding, truncation)
            
            if return_tensors == "pt":
                import torch
                return torch.tensor(token_ids)
            elif return_tensors == "tf":
                import tensorflow as tf
                return tf.constant(token_ids)
            else:
                return token_ids
        
        else:
            # Multiple texts
            results = []
            for t in text:
                result = self.encode(t, add_special_tokens, padding, truncation, max_length, stride, return_tensors, **kwargs)
                results.append(result)
            
            if return_tensors == "pt":
                import torch
                return torch.stack(results)
            elif return_tensors == "tf":
                import tensorflow as tf
                return tf.stack(results)
            else:
                return results
    
    def decode(
        self,
        token_ids: Union[int, List[int], torch.Tensor],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = True,
        **kwargs
    ) -> str:
        """Decode token IDs to text"""
        
        # Convert to list if needed
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        elif isinstance(token_ids, int):
            token_ids = [token_ids]
        
        # Convert IDs to tokens
        tokens = [self.ids_to_tokens.get(id, self.unk_token) for id in token_ids]
        
        # Remove special tokens if requested
        if skip_special_tokens:
            special_tokens = [self.unk_token, self.bos_token, self.eos_token, self.pad_token]
            if self.cls_token:
                special_tokens.append(self.cls_token)
            if self.sep_token:
                special_tokens.append(self.sep_token)
            if self.mask_token:
                special_tokens.append(self.mask_token)
            
            tokens = [token for token in tokens if token not in special_tokens]
        
        # Join tokens
        text = "".join(tokens)
        
        # Clean up spaces
        if clean_up_tokenization_spaces:
            text = " ".join(text.split())
        
        return text
    
    def _apply_padding_truncation(
        self,
        ids: List[int],
        max_length: Optional[int],
        padding: Union[bool, str],
        truncation: Union[bool, str]
    ) -> List[int]:
        """Apply padding and truncation to token IDs"""
        
        if max_length is None:
            max_length = self.model_max_length
        
        # Truncation
        if truncation and len(ids) > max_length:
            ids = ids[:max_length]
        
        # Padding
        if padding and len(ids) < max_length:
            pad_id = self.vocab[self.pad_token]
            ids.extend([pad_id] * (max_length - len(ids)))
        
        return ids
    
    def save_pretrained(self, save_directory: str, **kwargs):
        """Save tokenizer to directory"""
        os.makedirs(save_directory, exist_ok=True)
        
        # Save individual tokenizers
        if self.unigram_tokenizer.model_file:
            unigram_path = os.path.join(save_directory, "unigram.model")
            import shutil
            shutil.copy2(self.unigram_tokenizer.model_file, unigram_path)
        
        if self.bpe_tokenizer.model_file:
            bpe_path = os.path.join(save_directory, "bpe.model")
            import shutil
            shutil.copy2(self.bpe_tokenizer.model_file, bpe_path)
        
        # Save tokenizer configuration
        config = {
            "tokenizer_type": "hybrid",
            "vocab_size": self.vocab_size,
            "special_tokens": {
                "unk_token": self.unk_token,
                "bos_token": self.bos_token,
                "eos_token": self.eos_token,
                "pad_token": self.pad_token,
                "cls_token": self.cls_token,
                "sep_token": self.sep_token,
                "mask_token": self.mask_token,
            },
            "additional_special_tokens": self.additional_special_tokens,
            "vocab": self.vocab,
            "ids_to_tokens": self.ids_to_tokens,
        }
        
        config_path = os.path.join(save_directory, "tokenizer_config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        """Load tokenizer from pretrained directory"""
        
        # Load configuration
        config_path = os.path.join(pretrained_model_name_or_path, "tokenizer_config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
        else:
            config = {}
        
        # Find model files
        unigram_model_file = os.path.join(pretrained_model_name_or_path, "unigram.model")
        bpe_model_file = os.path.join(pretrained_model_name_or_path, "bpe.model")
        
        # Create tokenizer instance
        tokenizer = cls(
            unigram_model_file=unigram_model_file if os.path.exists(unigram_model_file) else None,
            bpe_model_file=bpe_model_file if os.path.exists(bpe_model_file) else None,
            **kwargs
        )
        
        # Load vocabulary if available
        if "vocab" in config:
            tokenizer.vocab = config["vocab"]
            tokenizer.ids_to_tokens = config["ids_to_tokens"]
            tokenizer.vocab_size = len(tokenizer.vocab)
        
        return tokenizer
    
    def __len__(self):
        """Get vocabulary size"""
        return self.vocab_size
    
    def __call__(
        self,
        text: Union[str, List[str]],
        text_pair: Optional[Union[str, List[str]]] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str] = False,
        truncation: Union[bool, str] = False,
        max_length: Optional[int] = None,
        stride: int = 0,
        return_tensors: Optional[str] = None,
        **kwargs
    ) -> BatchEncoding:
        """Call tokenizer on text"""
        
        # Handle text pairs (for BERT-style models)
        if text_pair is not None:
            if isinstance(text, str):
                text = [text]
            if isinstance(text_pair, str):
                text_pair = [text_pair]
            
            # Combine texts
            combined_texts = []
            for t1, t2 in zip(text, text_pair):
                combined_texts.append(f"{t1} {self.sep_token} {t2}" if self.sep_token else f"{t1} {t2}")
            text = combined_texts
        
        # Encode texts
        if isinstance(text, str):
            input_ids = self.encode(text, add_special_tokens, padding, truncation, max_length, stride, return_tensors, **kwargs)
            attention_mask = [1] * len(input_ids) if isinstance(input_ids, list) else None
        else:
            input_ids = []
            attention_masks = []
            for t in text:
                ids = self.encode(t, add_special_tokens, padding, truncation, max_length, stride, return_tensors, **kwargs)
                input_ids.append(ids)
                attention_masks.append([1] * len(ids) if isinstance(ids, list) else None)
            attention_mask = attention_masks
        
        # Create BatchEncoding
        encoding = BatchEncoding({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        })
        
        return encoding
