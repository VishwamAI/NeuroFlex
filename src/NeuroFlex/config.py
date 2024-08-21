from detectron2.config import get_cfg as detectron2_get_cfg

def get_cfg():
    """
    Wrapper function for Detectron2's get_cfg function.
    This allows for any additional custom configuration if needed.
    """
    cfg = detectron2_get_cfg()
    # Add any custom configuration here if needed
    return cfg
