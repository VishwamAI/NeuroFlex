import os
import logging
from typing import List, Optional

import sentencepiece

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Tokenizer:
    """
    A tokenizer class that uses SentencePiece for text encoding and decoding.

    This class provides methods for tokenizing text, encoding text to token IDs,
    decoding token IDs back to text, and handling special tokens.

    Attributes:
        sp_model (sentencepiece.SentencePieceProcessor): The SentencePiece model.
        n_words (int): The total number of tokens in the vocabulary.
        bos_id (int): The ID of the beginning-of-sentence token.
        eos_id (int): The ID of the end-of-sentence token.
        pad_id (int): The ID of the padding token.
        unk_id (int): The ID of the unknown token.
        space_id (int): The ID of the space token.
        special_chars (set): A set of special characters handled separately.

    Args:
        model_path (Optional[str]): The path to the SentencePiece model file.

    Raises:
        ValueError: If the model_path is invalid or the file doesn't exist.
        RuntimeError: If there's an error initializing the tokenizer.
    """

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

    def encode(self, s: str, bos: bool = True, eos: bool = True) -> List[int]:
        """
        Converts a string into a list of token IDs.

        This method encodes the input string using the SentencePiece model and handles special tokens.

        Args:
            s (str): The input string to be encoded.
            bos (bool): Whether to prepend the beginning-of-sentence token. Defaults to True.
            eos (bool): Whether to append the end-of-sentence token. Defaults to True.

        Returns:
            List[int]: A list of token IDs, including special tokens if specified.

        Raises:
            ValueError: If the input is not a string or if encoding fails.

        Special Token Handling:
            - BOS (Beginning of Sentence) token is added at the start if bos=True.
            - EOS (End of Sentence) token is added at the end if eos=True.
            - UNK (Unknown) tokens are replaced with the ID of '<unk>'.

        Example:
            >>> tokenizer = Tokenizer("path/to/model")
            >>> tokenizer.encode("Hello, world!")
            [1, 15, 12, 9, 8, 3, 2]  # Example output with BOS and EOS tokens

        Note:
            If the SentencePiece model fails to encode the string, a character-level
            fallback encoding is used, treating each character as a separate token.
        """
        if not isinstance(s, str):
            raise ValueError(f"Invalid input type: {type(s)}. Input must be a string.")

        tokens = []

        try:
            encoded = self.sp_model.EncodeAsIds(s)
            if not encoded:
                logging.warning(f"Failed to encode string: '{s}'")
                encoded = [self.sp_model.PieceToId(char) if char not in self.special_chars
                           else self.sp_model.PieceToId(f"▁{char}")
                           for char in s]

            if bos and self.bos_id != -1:
                tokens.append(self.bos_id)

            tokens.extend(encoded)

            if eos and self.eos_id != -1:
                tokens.append(self.eos_id)

        except Exception as e:
            logging.error(f"Error encoding '{s}': {str(e)}")
            raise ValueError(f"Encoding failed: {str(e)}")

        tokens = [t if t != self.unk_id else self.sp_model.PieceToId('<unk>') for t in tokens]

        logging.debug(f"Encoded '{s}' to {len(tokens)} tokens")
        return tokens

    def decode(self, t: List[int]) -> str:
        """
        Converts a list of token IDs into a string.

        This method takes a list of integer token IDs and decodes them back into a string.
        It handles special tokens (BOS, EOS, PAD) and performs post-processing on the decoded text.

        Args:
            t (List[int]): A list of integer token IDs to be decoded.

        Returns:
            str: The decoded string. If decoding fails, returns "[DECODING_FAILED]".

        Raises:
            No exceptions are raised. Errors are logged and a failure string is returned.

        Note:
            - Special tokens (BOS, EOS, PAD) are removed before decoding.
            - If the initial decoding results in an empty string, a fallback method is used.
            - Extensive logging is performed for debugging purposes.

        Example:
            >>> tokenizer = Tokenizer("model_path")
            >>> tokens = [1, 100, 200, 300, 2]  # Assuming 1 is BOS and 2 is EOS
            >>> decoded_text = tokenizer.decode(tokens)
            >>> print(decoded_text)
            "Hello world"
        """
        if not isinstance(t, list):
            logging.warning(f"Invalid input type for decoding: {type(t)}")
            return "[DECODING_FAILED]"

        if not all(isinstance(token, int) for token in t):
            logging.warning("Invalid token type in input list")
            return "[DECODING_FAILED]"

        try:
            original_t = t
            logging.debug(f"Original tokens: {original_t}")

            # Handle special tokens but keep invalid integers
            t = [token for token in t if token not in {self.bos_id, self.eos_id, self.pad_id}]
            logging.debug(f"Tokens after handling special tokens: {t}")

            # Call DecodeIds even for empty list
            decoded_text = self.sp_model.DecodeIds(t)
            logging.debug(f"Raw decoded text: {decoded_text}")

            decoded_text = self._post_process_decoded_text(decoded_text)
            logging.debug(f"Post-processed decoded text: {decoded_text}")

            if not decoded_text:
                logging.warning("Decoding resulted in empty string. Using fallback method.")
                decoded_text = self._fallback_decode(t)
                logging.debug(f"Fallback decoded text: {decoded_text}")

            if not decoded_text:
                logging.warning("Decoding failed to produce any text")
                return "[DECODING_FAILED]"

            logging.debug(f"Final decoded text: '{decoded_text}'")
            return decoded_text
        except Exception as e:
            logging.error(f"Error during decoding: {str(e)}")
            logging.debug(f"Problematic tokens: {t}")
            return "[DECODING_FAILED]"

    def _fallback_decode(self, t: List[int]) -> str:
        """
        Fallback method for decoding when the main method fails.

        This method attempts to decode the input token list by converting each token
        to its corresponding piece (string representation) and joining them together.
        It skips any unknown tokens (UNK) during this process.

        Args:
            t (List[int]): A list of token IDs to be decoded.

        Returns:
            str: The decoded string, or an empty string if decoding fails.

        Note:
            This method is used as a last resort when the primary decoding method fails.
            It may not preserve the exact original text structure but aims to recover
            as much information as possible from the token IDs.
        """
        try:
            return ''.join([self.sp_model.IdToPiece(token) for token in t if token != self.unk_id])
        except Exception as e:
            logging.error(f"Fallback decoding failed: {str(e)}")
            return ""

    def _handle_special_tokens(self, tokens: List[int]) -> List[int]:
        """
        Handles special tokens like BOS (Beginning of Sentence), EOS (End of Sentence), and PAD (Padding).

        This method filters out special tokens from the input token list. It's typically used
        during decoding to remove tokens that don't contribute to the actual content.

        Args:
            tokens (List[int]): A list of token IDs, potentially including special tokens.

        Returns:
            List[int]: A filtered list of token IDs with special tokens removed.

        Note:
            - BOS, EOS, and PAD tokens are removed.
            - Any token with ID -1 is also removed, as this is often used to represent invalid or placeholder tokens.
        """
        return [token for token in tokens if token not in {self.bos_id, self.eos_id, self.pad_id} and token != -1]

    def _post_process_decoded_text(self, text: str) -> str:
        """
        Post-processes the decoded text to improve readability.

        This method performs two main operations:
        1. Removes spaces before punctuation marks.
        2. Normalizes whitespace by removing extra spaces and stripping leading/trailing spaces.

        Args:
            text (str): The raw decoded text.

        Returns:
            str: The post-processed text with improved readability.

        Example:
            >>> tokenizer._post_process_decoded_text("Hello ,  world !")
            "Hello, world!"
        """
        for punct in self.special_chars:
            text = text.replace(f' {punct}', punct)
        return ' '.join(text.split()).strip()

    def _split_text(self, text: str) -> List[str]:
        """
        Splits text into parts, preserving special characters and whitespace.

        This method is used internally to break down the input text into smaller parts
        while maintaining the integrity of special characters and whitespace. This is
        crucial for accurate tokenization and detokenization.

        Args:
            text (str): The input text to be split.

        Returns:
            List[str]: A list of string parts, where each part is either:
                       - A contiguous sequence of non-special, non-whitespace characters
                       - A single special character
                       - A single whitespace character

        Example:
            >>> tokenizer._split_text("Hello, world!")
            ['Hello', ',', ' ', 'world', '!']

        Note:
            Special characters are defined in self.special_chars.
            Whitespace is determined using the str.isspace() method.
        """
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
        """
        Tokenizes the input text into a list of token strings.

        This method splits the input text into individual tokens using the SentencePiece model.
        It removes any leading whitespace characters from each token and filters out empty tokens.

        Args:
            text (str): The input text to be tokenized.

        Returns:
            List[str]: A list of token strings.

        Raises:
            ValueError: If the input is not a string.

        Example:
            >>> tokenizer = Tokenizer("path/to/model")
            >>> tokenizer.tokenize("Hello, world!")
            ['Hello', ',', 'world', '!']
        """
        if not isinstance(text, str):
            raise ValueError(f"Invalid input type for tokenization: {type(text)}. Expected string.")
        try:
            tokens = self.sp_model.EncodeAsPieces(text)
            return [token.lstrip('▁') for token in tokens if token != '▁']
        except Exception as e:
            logging.error(f"Error tokenizing '{text}': {str(e)}")
            return []

    def detokenize(self, tokens: List[str]) -> str:
        """
        Converts a list of token strings back into text.

        Args:
            tokens (List[str]): A list of token strings to be detokenized.

        Returns:
            str: The detokenized text.

        Raises:
            ValueError: If the input is not a list of strings.

        Note:
            This method uses the SentencePiece model's DecodePieces function to convert
            tokens back into text. It then applies post-processing to improve readability.

        Example:
            >>> tokenizer = Tokenizer("path/to/model")
            >>> tokens = ["Hello", "world", "!"]
            >>> tokenizer.detokenize(tokens)
            'Hello world!'
        """
        if not isinstance(tokens, list) or not all(isinstance(token, str) for token in tokens):
            error_msg = f"Invalid input for detokenization: {tokens}. Expected a list of strings."
            logging.warning(error_msg)
            raise ValueError(error_msg)
        try:
            text = self.sp_model.DecodePieces(tokens)
            return self._post_process_decoded_text(text)
        except Exception as e:
            logging.error(f"Error detokenizing tokens: {str(e)}")
            return "[DETOKENIZATION_FAILED]"
