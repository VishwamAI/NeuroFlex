# from detectron2.config import get_cfg as detectron2_get_cfg

def get_cfg():
    """
    Mock function to replace Detectron2's get_cfg function.
    This returns a simple dictionary as a mock configuration.
    """
    cfg = {
        "DATASETS": {"TRAIN": (), "TEST": ()},
        "DATALOADER": {"NUM_WORKERS": 2},
        "MODEL": {"WEIGHTS": "", "ROI_HEADS": {"BATCH_SIZE_PER_IMAGE": 512, "NUM_CLASSES": 0}},
        "SOLVER": {"IMS_PER_BATCH": 2, "BASE_LR": 0.00025, "MAX_ITER": 1000},
        "OUTPUT_DIR": ""
    }
    return cfg
