import logging
from typing import List, Dict, Optional
from functools import lru_cache
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Tokenizer:
    def __init__(self, model_name: str = "bert-base-uncased", use_fast: bool = True):
        self.tokenizers = {}
        self.special_tokens = {'<unk>': 0, '<s>': 1, '</s>': 2, '<pad>': 3}
        self.cache = {}
        try:
            self.load_tokenizer(model_name, use_fast)
        except Exception as e:
            logger.error(f"Error loading tokenizer: {str(e)}")
            raise

    def load_tokenizer(self, model_name: str, use_fast: bool = True):
        self.tokenizers['default'] = AutoTokenizer.from_pretrained(model_name, use_fast=use_fast)
        logger.info(f"Loaded tokenizer: {model_name}")

    def load_pretrained_tokenizers(self, model_name: str = "bert-base-multilingual-cased"):
        """
        Load pre-trained tokenizers for multiple languages using AutoTokenizer.
        """
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            logger.info(f"Loaded pre-trained tokenizer: {model_name}")
        except Exception as e:
            logger.error(f"Error loading pre-trained tokenizer: {str(e)}")
            raise

    def add_special_tokens(self, special_tokens: List[str]):
        """
        Add new special tokens to the tokenizer.
        """
        try:
            new_tokens = [token for token in special_tokens if token not in self.tokenizer.special_tokens_map.values()]
            if new_tokens:
                self.tokenizer.add_special_tokens({'additional_special_tokens': new_tokens})
                logger.info(f"Added {len(new_tokens)} new special tokens.")
            else:
                logger.info("No new special tokens to add.")
        except Exception as e:
            logger.error(f"Error adding special tokens: {str(e)}")
            raise

    # The load_model method has been removed as it's no longer needed with the current implementation.
    # If you need to load a specific model, use the load_tokenizer method instead.

    @lru_cache(maxsize=100000)
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Convert text to a list of token ids using AutoTokenizer.
        """
        if not isinstance(text, str):
            raise ValueError("Input must be a string")
        try:
            if 'default' not in self.tokenizers:
                raise ValueError("Tokenizer not loaded. Call load_tokenizer() first.")

            return self.tokenizers['default'].encode(text, add_special_tokens=add_special_tokens)
        except Exception as e:
            logger.error(f"Error encoding text: {str(e)}")
            raise

    def decode(self, token_ids: Optional[List[int]], skip_special_tokens: bool = True) -> str:
        """
        Convert a list of token ids back to text using AutoTokenizer.
        """
        if token_ids is None:
            raise ValueError("Input token_ids cannot be None")
        if not token_ids:
            return ""
        try:
            if 'default' not in self.tokenizers:
                raise ValueError("Tokenizer not loaded. Call load_tokenizer() first.")
            if not all(isinstance(token, int) for token in token_ids):
                raise ValueError("All token ids must be integers.")
            decoded_text = self.tokenizers['default'].decode(token_ids, skip_special_tokens=skip_special_tokens)
            return decoded_text if decoded_text else ""
        except Exception as e:
            logger.error(f"Error decoding token ids: {str(e)}")
            raise

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into subwords using AutoTokenizer.
        """
        if 'default' not in self.tokenizers:
            raise ValueError("Tokenizer not loaded. Call load_tokenizer() first.")
        return self.tokenizers['default'].tokenize(text)

    def get_vocab(self) -> Dict[str, int]:
        """
        Get the vocabulary as a dictionary of token to id.
        """
        if 'default' not in self.tokenizers:
            raise ValueError("Tokenizer not loaded. Call load_tokenizer() first.")
        return self.tokenizers['default'].get_vocab()

    def get_vocab_size(self) -> int:
        """
        Get the size of the vocabulary.
        """
        if 'default' not in self.tokenizers:
            raise ValueError("Tokenizer not loaded. Call load_tokenizer() first.")
        return len(self.tokenizers['default'].get_vocab())

    def token_to_id(self, token: str) -> int:
        """
        Convert a token to its id.
        """
        if 'default' not in self.tokenizers:
            raise ValueError("Tokenizer not loaded. Call load_tokenizer() first.")
        return self.tokenizers['default'].convert_tokens_to_ids(token)

    def id_to_token(self, id: int) -> str:
        """
        Convert an id to its token.
        """
        if 'default' not in self.tokenizers:
            raise ValueError("Tokenizer not loaded. Call load_tokenizer() first.")
        return self.tokenizers['default'].convert_ids_to_tokens(id)

# This method has been removed as it's already implemented earlier in the file.
# The earlier implementation uses the AutoTokenizer's built-in method for adding special tokens.

# The get_tokenizer function and clean_text function have been removed
# as they are no longer used in the current implementation.
