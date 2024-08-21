from gramformer import Gramformer
import torch

def correct_grammar(text: str) -> str:
    """
    Correct the grammar of the input text using Gramformer.

    Args:
        text (str): The input text to be corrected.

    Returns:
        str: The corrected text.
    """
    # Set up Gramformer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gf = Gramformer(models=1, use_gpu=device=="cuda")  # 1 = corrector model

    # Correct the text
    corrected_texts = list(gf.correct(text, max_candidates=1))

    # Return the first correction if available, otherwise return the original text
    return corrected_texts[0] if corrected_texts else text
