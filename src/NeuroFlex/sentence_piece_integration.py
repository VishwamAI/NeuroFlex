import sentencepiece as spm
import jax.numpy as jnp
from typing import List, Dict, Any, Optional
import logging

class SentencePieceIntegration:
    def __init__(self, model_path: str):
        try:
            self.sp = spm.SentencePieceProcessor()
            self.sp.Load(model_path)
            logging.info(f"SentencePiece model loaded from {model_path}")
        except Exception as e:
            logging.error(f"Error loading SentencePiece model: {str(e)}")
            raise

    def encode(self, text: str) -> List[int]:
        try:
            return self.sp.EncodeAsIds(text)
        except Exception as e:
            logging.error(f"Error encoding text: {str(e)}")
            return []

    def decode(self, ids: List[int]) -> str:
        try:
            return self.sp.DecodeIds(ids)
        except Exception as e:
            logging.error(f"Error decoding ids: {str(e)}")
            return ""

    def tokenize(self, text: str) -> List[str]:
        try:
            return self.sp.EncodeAsPieces(text)
        except Exception as e:
            logging.error(f"Error tokenizing text: {str(e)}")
            return []

    def detokenize(self, tokens: List[str]) -> str:
        try:
            return self.sp.DecodePieces(tokens)
        except Exception as e:
            logging.error(f"Error detokenizing tokens: {str(e)}")
            return ""

    def get_vocab_size(self) -> int:
        return self.sp.GetPieceSize()

    def id_to_piece(self, id: int) -> str:
        return self.sp.IdToPiece(id)

    def piece_to_id(self, piece: str) -> int:
        return self.sp.PieceToId(piece)

def create_sentence_piece_integration(model_path: str) -> SentencePieceIntegration:
    return SentencePieceIntegration(model_path)

# Example usage
if __name__ == "__main__":
    model_path = "path/to/your/sentencepiece.model"
    sp_integration = create_sentence_piece_integration(model_path)

    text = "Hello, this is a test sentence."
    encoded = sp_integration.encode(text)
    decoded = sp_integration.decode(encoded)
    tokens = sp_integration.tokenize(text)
    detokenized = sp_integration.detokenize(tokens)

    print(f"Original text: {text}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")
    print(f"Tokens: {tokens}")
    print(f"Detokenized: {detokenized}")
    print(f"Vocabulary size: {sp_integration.get_vocab_size()}")
