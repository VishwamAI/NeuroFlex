import os
import logging
from typing import List, Optional

import sentencepiece

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Tokenizer:

    def __init__(self, model_path: Optional[str]):
        if not model_path or not os.path.isfile(model_path):
            raise ValueError(f"Invalid model path: {model_path}")
        try:
            self.sp_model = sentencepiece.SentencePieceProcessor()
            self.sp_model.Load(model_path)
            self.n_words: int = self.sp_model.GetPieceSize()
            self.bos_id: int = self.sp_model.bos_id()
            self.eos_id: int = self.sp_model.eos_id()
            self.pad_id: int = self.sp_model.pad_id()
            self.unk_id: int = self.sp_model.unk_id()
            self.space_id: int = self.sp_model.PieceToId('<space>')
            self.special_chars = set('.,!?;:()[]{}""''')
            logging.info(f"Tokenizer initialized successfully with {self.n_words} tokens")
        except Exception as e:
            logging.error(f"Error initializing tokenizer: {str(e)}")
            raise RuntimeError(f"Tokenizer initialization failed: {str(e)}")

    def encode(self, s: str, bos: bool = True, eos: bool = False) -> List[int]:
        """
        Converts a string into a list of tokens.

        Args:
            s (str): The input string to be encoded.
            bos (bool): Whether to prepend the beginning-of-sentence token. Defaults to True.
            eos (bool): Whether to append the end-of-sentence token. Defaults to False.

        Returns:
            List[int]: A list of token IDs.

        Raises:
            ValueError: If the input is not a string or if encoding fails.
        """
        if not isinstance(s, str):
            raise ValueError(f"Invalid input type: {type(s)}. Input must be a string.")

        tokens = []
        if bos and self.bos_id != -1:
            tokens.append(self.bos_id)

        if s:
            try:
                parts = self._split_text(s)
                for part in parts:
                    if part.strip() or part in self.special_chars:
                        if part in self.special_chars:
                            tokens.append(self.sp_model.PieceToId(f"▁{part}"))
                        else:
                            encoded = self.sp_model.EncodeAsIds(part)
                            if not encoded:
                                logging.warning(f"Failed to encode part: '{part}'")
                                encoded = [self.sp_model.PieceToId(char) if char not in self.special_chars
                                           else self.sp_model.PieceToId(f"▁{char}")
                                           for char in part]
                            tokens.extend(encoded)
                    elif part.isspace():
                        tokens.append(self.space_id)
            except Exception as e:
                logging.error(f"Error encoding '{s}': {str(e)}")
                raise ValueError(f"Encoding failed: {str(e)}")
        elif not bos and not eos:
            logging.debug("Empty input with no BOS/EOS tokens requested, returning empty list")
            return []

        if eos and self.eos_id != -1:
            tokens.append(self.eos_id)

        tokens = [t if t != self.unk_id else self.sp_model.PieceToId('<unk>') for t in tokens]

        logging.debug(f"Encoded '{s}' to {len(tokens)} tokens")
        return tokens

    def decode(self, t: List[int]) -> str:
        """Converts a list of tokens into a string."""
        if not isinstance(t, list) or not t or not all(isinstance(token, int) for token in t):
            logging.warning(f"Invalid input for decoding: {t}")
            return ""

        try:
            t = self._handle_special_tokens(t)
            if not t:
                return ""
            decoded_text = self.sp_model.DecodeIds(t)
            decoded_text = self._post_process_decoded_text(decoded_text)
            if not decoded_text:
                logging.warning("Decoding resulted in empty string. Using fallback method.")
                decoded_text = self._fallback_decode(t)
            if not decoded_text:
                logging.warning("Fallback decoding also resulted in empty string.")
                return "[DECODING_FAILED]"
            logging.debug(f"Decoded {len(t)} tokens to: '{decoded_text}'")
            return decoded_text
        except Exception as e:
            logging.error(f"Error during decoding: {str(e)}")
            logging.debug(f"Problematic tokens: {t}")
            return self._fallback_decode(t) or "[DECODING_FAILED]"

    def _fallback_decode(self, t: List[int]) -> str:
        """Fallback method for decoding when the main method fails."""
        try:
            return ''.join([self.sp_model.IdToPiece(token) for token in t if token != self.unk_id])
        except Exception as e:
            logging.error(f"Fallback decoding failed: {str(e)}")
            return ""

    def _handle_special_tokens(self, tokens: List[int]) -> List[int]:
        """Handles special tokens like BOS, EOS, and PAD."""
        return [token for token in tokens if token not in {self.bos_id, self.eos_id, self.pad_id} and token != -1]

    def _post_process_decoded_text(self, text: str) -> str:
        """Post-processes the decoded text to improve readability."""
        for punct in self.special_chars:
            text = text.replace(f' {punct}', punct)
        return ' '.join(text.split()).strip()

    def _split_text(self, text: str) -> List[str]:
        """Splits text into parts, preserving special characters and whitespace."""
        parts = []
        current_part = ""
        for char in text:
            if char.isspace() or char in self.special_chars:
                if current_part:
                    parts.append(current_part)
                    current_part = ""
                parts.append(char)
            else:
                current_part += char
        if current_part:
            parts.append(current_part)
        return parts

    def tokenize(self, text: str) -> List[str]:
        """Tokenizes the input text into a list of token strings."""
        if not isinstance(text, str):
            logging.warning(f"Invalid input type for tokenization: {type(text)}. Expected string.")
            return []
        try:
            tokens = self.sp_model.EncodeAsPieces(text)
            return [token if token != '▁' else ' ' for token in tokens]
        except Exception as e:
            logging.error(f"Error tokenizing '{text}': {str(e)}")
            return []

    def detokenize(self, tokens: List[str]) -> str:
        """Converts a list of token strings back into text."""
        if not isinstance(tokens, list) or not all(isinstance(token, str) for token in tokens):
            logging.warning(f"Invalid input for detokenization: {tokens}")
            return ""
        try:
            text = self.sp_model.DecodePieces(tokens)
            return self._post_process_decoded_text(text)
        except Exception as e:
            logging.error(f"Error detokenizing tokens: {str(e)}")
            return ""
