from gramformer import Gramformer

def correct_grammar(text):
    """
    Correct the grammar of the input text using Gramformer.

    Args:
    text (str): The input text to be corrected.

    Returns:
    str: The corrected text.
    """
    gf = Gramformer(models=1, use_gpu=False)  # Initialize Gramformer
    corrected = gf.correct(text)
    return next(iter(corrected), text)  # Return the first correction or the original text if no corrections
