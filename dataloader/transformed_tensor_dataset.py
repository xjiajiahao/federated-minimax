from torch import Tensor
from torch.utils.data import Dataset
from typing import Tuple, Callable, Optional

class TransformedTensorDataset(Dataset[Tuple[Tensor, ...]]):
    r"""Dataset wrapping tensors.

    Each sample will be retrieved by indexing tensors along the first dimension.

    Args:
        *tensors (Tensor): tensors that have the same size of the first dimension.
    """
    tensors: Tuple[Tensor, ...]

    def __init__(self, *tensors: Tensor, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None) -> None:
        assert len(tensors) == 2 and all(tensors[0].size(0) == tensor.size(0) for tensor in tensors), "Size mismatch between tensors"
        self.tensors = tensors
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img, target = self.tensors[0][index], self.tensors[1][index]
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return self.tensors[0].size(0)
