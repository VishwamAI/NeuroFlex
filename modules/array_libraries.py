import numpy as np
import torch

class ArrayLibraries:
    @staticmethod
    def convert_numpy_to_pytorch(np_array):
        return torch.from_numpy(np_array)

    @staticmethod
    def convert_pytorch_to_numpy(torch_tensor):
        return torch_tensor.detach().cpu().numpy()
