from transformers.tokenization_utils_base import BatchEncoding

from collections.abc import Mapping, Sequence

import torch.utils.data
from torch.utils.data.dataloader import default_collate

from torch_geometric.data import Data, Batch
import numpy as np

# Input can be tensors organized recursively by list, tuples or dict.
def basic_collate(batch):
    elem = batch[0]
    if isinstance(elem, Data):
        return Batch.from_data_list(batch)
    elif isinstance(elem, torch.Tensor):
        return default_collate(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, Mapping):
        return {key: basic_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):
        return type(elem)(*(basic_collate(s) for s in zip(*batch)))
    elif isinstance(elem, Sequence) and not isinstance(elem, str):
        return [basic_collate(s) for s in zip(*batch)]

    raise TypeError('DataLoader found invalid type: {}'.format(type(elem)))
