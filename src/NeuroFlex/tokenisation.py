import nltk
from nltk.tokenize import word_tokenize
from typing import List

# Download the necessary NLTK data
nltk.download('punkt', quiet=True)

def tokenize_text(text: str) -> List[str]:
    """
    Tokenize the input text using NLTK's word_tokenize function.

    Args:
        text (str): The input text to be tokenized.

    Returns:
        List[str]: A list of tokens.
    """
    return word_tokenize(text)
