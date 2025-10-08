"""
Custom tokenizer wrapper for Hugging Face integration
"""

import os
import json
import sentencepiece as spm
from typing import List, Optional, Union, Dict, Any, Tuple
from transformers import PreTrainedTokenizer
from transformers.tokenization_utils_base import BatchEncoding


class CustomTokenizer(PreTrainedTokenizer):
    """
    Custom tokenizer based on SentencePiece with Hugging Face integration
    """
    
    def __init__(
        self,
        model_file: Optional[str] = None,
        vocab_file: Optional[str] = None,
        tokenizer_config: Optional[Dict[str, Any]] = None,
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
        """
        Initialize custom tokenizer
        
        Args:
            model_file: Path to SentencePiece model file
            vocab_file: Path to vocabulary file
            tokenizer_config: Tokenizer configuration dictionary
            unk_token: Unknown token
            bos_token: Beginning of sequence token
            eos_token: End of sequence token
            pad_token: Padding token
            cls_token: Classification token (for BERT-style models)
            sep_token: Separator token (for BERT-style models)
            mask_token: Mask token (for BERT-style models)
            additional_special_tokens: Additional special tokens
            **kwargs: Additional arguments
        """
        
        # Initialize SentencePiece processor
        self.sp = spm.SentencePieceProcessor()
        
        # Load model if provided
        if model_file and os.path.exists(model_file):
            self.sp.load(model_file)
            self.model_file = model_file
        else:
            self.model_file = None
        
        # Set special tokens
        self.unk_token = unk_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.cls_token = cls_token
        self.sep_token = sep_token
        self.mask_token = mask_token
        
        # Additional special tokens
        if additional_special_tokens is None:
            additional_special_tokens = []
        self.additional_special_tokens = additional_special_tokens
        
        # Get vocabulary size
        if self.model_file:
            self.vocab_size = self.sp.get_piece_size()
        else:
            self.vocab_size = 32000  # Default
        
        # Load configuration if provided
        if tokenizer_config:
            self._load_config(tokenizer_config)
        
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
    
    def _load_config(self, config: Dict[str, Any]):
        """Load configuration from dictionary"""
        if "special_tokens" in config:
            special_tokens = config["special_tokens"]
            self.unk_token = special_tokens.get("unk_token", self.unk_token)
            self.bos_token = special_tokens.get("bos_token", self.bos_token)
            self.eos_token = special_tokens.get("eos_token", self.eos_token)
            self.pad_token = special_tokens.get("pad_token", self.pad_token)
        
        if "token_ids" in config:
            token_ids = config["token_ids"]
            # Update token IDs if needed
            pass
    
    @property
    def vocab_size(self) -> int:
        """Get vocabulary size"""
        return self.sp.get_piece_size()
    
    def get_vocab(self) -> Dict[str, int]:
        """Get vocabulary mapping"""
        vocab = {}
        for i in range(self.vocab_size):
            piece = self.sp.id_to_piece(i)
            vocab[piece] = i
        return vocab
    
    def _tokenize(self, text: str, **kwargs) -> List[str]:
        """Tokenize text into tokens"""
        if not self.model_file:
            raise ValueError("Tokenizer not initialized. Please load a model file.")
        
        return self.sp.encode(text, out_type=str)
    
    def _convert_token_to_id(self, token: str) -> int:
        """Convert token to ID"""
        if not self.model_file:
            raise ValueError("Tokenizer not initialized. Please load a model file.")
        
        return self.sp.piece_to_id(token)
    
    def _convert_id_to_token(self, index: int) -> str:
        """Convert ID to token"""
        if not self.model_file:
            raise ValueError("Tokenizer not initialized. Please load a model file.")
        
        return self.sp.id_to_piece(index)
    
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
            tokens = self._tokenize(text)
            if add_special_tokens:
                tokens = [self.bos_token] + tokens + [self.eos_token]
            
            ids = self.convert_tokens_to_ids(tokens)
            
            # Apply padding/truncation
            if padding or truncation:
                ids = self._apply_padding_truncation(ids, max_length, padding, truncation)
            
            if return_tensors == "pt":
                import torch
                return torch.tensor(ids)
            elif return_tensors == "tf":
                import tensorflow as tf
                return tf.constant(ids)
            else:
                return ids
        
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
        
        if not self.model_file:
            raise ValueError("Tokenizer not initialized. Please load a model file.")
        
        # Convert to list if needed
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        elif isinstance(token_ids, int):
            token_ids = [token_ids]
        
        # Decode using SentencePiece
        text = self.sp.decode(token_ids)
        
        # Clean up special tokens if requested
        if skip_special_tokens:
            special_tokens = [self.unk_token, self.bos_token, self.eos_token, self.pad_token]
            if self.cls_token:
                special_tokens.append(self.cls_token)
            if self.sep_token:
                special_tokens.append(self.sep_token)
            if self.mask_token:
                special_tokens.append(self.mask_token)
            
            for token in special_tokens:
                text = text.replace(token, "")
        
        # Clean up tokenization spaces
        if clean_up_tokenization_spaces:
            text = self._clean_up_tokenization_spaces(text)
        
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
            pad_id = self._convert_token_to_id(self.pad_token)
            ids.extend([pad_id] * (max_length - len(ids)))
        
        return ids
    
    def _clean_up_tokenization_spaces(self, text: str) -> str:
        """Clean up tokenization spaces"""
        # Remove extra spaces
        text = " ".join(text.split())
        return text
    
    def save_pretrained(self, save_directory: str, **kwargs):
        """Save tokenizer to directory"""
        import shutil
        
        os.makedirs(save_directory, exist_ok=True)
        
        # Save SentencePiece model
        if self.model_file:
            model_filename = os.path.basename(self.model_file)
            shutil.copy2(self.model_file, os.path.join(save_directory, model_filename))
        
        # Save tokenizer configuration
        config = {
            "model_file": self.model_file,
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
        }
        
        config_path = os.path.join(save_directory, "tokenizer_config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        # Save tokenizer.json for Hugging Face compatibility
        self._save_tokenizer_json(save_directory)
    
    def _save_tokenizer_json(self, save_directory: str):
        """Save tokenizer in Hugging Face JSON format"""
        import json
        
        # Create tokenizer.json compatible format
        tokenizer_json = {
            "version": "1.0",
            "truncation": None,
            "padding": None,
            "added_tokens": [],
            "normalizer": None,
            "pre_tokenizer": None,
            "post_processor": None,
            "decoder": None,
            "model": {
                "type": "SentencePiece",
                "vocab": self.get_vocab(),
                "unk_id": self._convert_token_to_id(self.unk_token),
                "bos_id": self._convert_token_to_id(self.bos_token),
                "eos_id": self._convert_token_to_id(self.eos_token),
                "pad_id": self._convert_token_to_id(self.pad_token),
            }
        }
        
        json_path = os.path.join(save_directory, "tokenizer.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(tokenizer_json, f, indent=2, ensure_ascii=False)
    
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
        
        # Find model file
        model_file = config.get("model_file")
        if not model_file or not os.path.exists(model_file):
            # Look for .model files in directory
            for file in os.listdir(pretrained_model_name_or_path):
                if file.endswith('.model'):
                    model_file = os.path.join(pretrained_model_name_or_path, file)
                    break
        
        if not model_file or not os.path.exists(model_file):
            raise FileNotFoundError(f"Model file not found in {pretrained_model_name_or_path}")
        
        # Create tokenizer instance
        tokenizer = cls(
            model_file=model_file,
            tokenizer_config=config,
            **kwargs
        )
        
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
