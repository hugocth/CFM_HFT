import torch, math, warnings
from torch.utils.data import Dataset, DataLoader

from torch import default_generator, randperm
from torch._utils import _accumulate
from torch.utils.data.dataset import Subset

from typing import List

import random

class TestDataset(Dataset):

    def __init__(self, x_test, x_cols):
        super().__init__()
        self.x_test = torch.tensor(x_test[x_cols].values.reshape(-1,100,len(x_cols)), dtype=torch.float32)

    def __len__(self):
        return len(self.x_test)
    
    def __getitem__(self, idx):
        return self.x_test[idx]
    
class ObsDataset(Dataset):

    def __init__(self, X_train, y_train, x_cols):
        super().__init__()
        self.X_train = torch.tensor(X_train[x_cols].values.reshape(-1,100,len(x_cols)), dtype=torch.float32)
        self.y_train = torch.tensor(y_train.values[:,1], dtype=torch.long)

    def __len__(self):
        return len(self.X_train)
    
    def __getitem__(self, idx):
        return self.X_train[idx], self.y_train[idx]
    
# class ObsDataset(Dataset):

#     def __init__(self, X_train, y_train, x_cols):
#         super().__init__()
#         self.X_train = X_train
#         self.y_train = y_train
#         self.x_cols = x_cols

#     def __len__(self):
#         return len(self.X_train)//100
    
#     def __getitem__(self, idx):
#         obs = torch.tensor(self.X_train.query(f"obs_id=={idx}")[self.x_cols].values, dtype=torch.float32).to(device)
#         label = self.y_train.query(f"obs_id=={idx}")["eqt_code_cat"].item()
#         label = torch.tensor(label, dtype=torch.long).to(device)
#         return obs, label
    
def get_dataloaders(X_train, y_train, x_cols, batch_size=512):
    full_dataset = ObsDataset(X_train, y_train, x_cols)
    train_dataset, test_dataset = random_split(full_dataset, [0.8,0.2])
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=min(batch_size*4, len(test_dataset)), drop_last=True)
    return (train_dataloader, test_dataloader)

def random_split(dataset, lengths,
                 generator=default_generator):
    r"""
    Randomly split a dataset into non-overlapping new datasets of given lengths.

    If a list of fractions that sum up to 1 is given,
    the lengths will be computed automatically as
    floor(frac * len(dataset)) for each fraction provided.

    After computing the lengths, if there are any remainders, 1 count will be
    distributed in round-robin fashion to the lengths
    until there are no remainders left.

    Optionally fix the generator for reproducible results, e.g.:

    >>> random_split(range(10), [3, 7], generator=torch.Generator().manual_seed(42))
    >>> random_split(range(30), [0.3, 0.3, 0.4], generator=torch.Generator(
    ...   ).manual_seed(42))

    Args:
        dataset (Dataset): Dataset to be split
        lengths (sequence): lengths or fractions of splits to be produced
        generator (Generator): Generator used for the random permutation.
    """
    if math.isclose(sum(lengths), 1) and sum(lengths) <= 1:
        subset_lengths: List[int] = []
        for i, frac in enumerate(lengths):
            if frac < 0 or frac > 1:
                raise ValueError(f"Fraction at index {i} is not between 0 and 1")
            n_items_in_split = int(
                math.floor(len(dataset) * frac)  # type: ignore[arg-type]
            )
            subset_lengths.append(n_items_in_split)
        remainder = len(dataset) - sum(subset_lengths)  # type: ignore[arg-type]
        # add 1 to all the lengths in round-robin fashion until the remainder is 0
        for i in range(remainder):
            idx_to_add_at = i % len(subset_lengths)
            subset_lengths[idx_to_add_at] += 1
        lengths = subset_lengths
        for i, length in enumerate(lengths):
            if length == 0:
                warnings.warn(f"Length of split at index {i} is 0. "
                              f"This might result in an empty dataset.")

    # Cannot verify that dataset is Sized
    if sum(lengths) != len(dataset):    # type: ignore[arg-type]
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

    indices = randperm(sum(lengths), generator=generator).tolist()  # type: ignore[call-overload]
    return [Subset(dataset, indices[offset - length : offset]) for offset, length in zip(_accumulate(lengths), lengths)]


def random_k_fold(dataset, lengths,
                 generator=default_generator):
    r"""
    Randomly split a dataset into k (train_subset, test_subset) folds.

    If a list of fractions that sum up to 1 is given,
    the lengths will be computed automatically as
    floor(frac * len(dataset)) for each fraction provided.

    After computing the lengths, if there are any remainders, 1 count will be
    distributed in round-robin fashion to the lengths
    until there are no remainders left.

    Optionally fix the generator for reproducible results, e.g.:

    >>> random_split(range(10), [3, 7], generator=torch.Generator().manual_seed(42))
    >>> random_split(range(30), [0.3, 0.3, 0.4], generator=torch.Generator(
    ...   ).manual_seed(42))

    Args:
        dataset (Dataset): Dataset to be split
        lengths (sequence): lengths or fractions of splits to be produced
        generator (Generator): Generator used for the random permutation.
    """
    if math.isclose(sum(lengths), 1) and sum(lengths) <= 1:
        subset_lengths: List[int] = []
        for i, frac in enumerate(lengths):
            if frac < 0 or frac > 1:
                raise ValueError(f"Fraction at index {i} is not between 0 and 1")
            n_items_in_split = int(
                math.floor(len(dataset) * frac)  # type: ignore[arg-type]
            )
            subset_lengths.append(n_items_in_split)
        remainder = len(dataset) - sum(subset_lengths)  # type: ignore[arg-type]
        # add 1 to all the lengths in round-robin fashion until the remainder is 0
        for i in range(remainder):
            idx_to_add_at = i % len(subset_lengths)
            subset_lengths[idx_to_add_at] += 1
        lengths = subset_lengths
        for i, length in enumerate(lengths):
            if length == 0:
                warnings.warn(f"Length of split at index {i} is 0. "
                              f"This might result in an empty dataset.")

    # Cannot verify that dataset is Sized
    if sum(lengths) != len(dataset):    # type: ignore[arg-type]
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

    indices = randperm(sum(lengths), generator=generator).tolist()  # type: ignore[call-overload]
    idxs_val = [indices[offset - length : offset] for offset, length in zip(_accumulate(lengths), lengths)]
    idxs_train = [list(set(range(len(dataset))) - set(idx_val)) for idx_val in idxs_val]
    return [(Subset(dataset, idx_train), Subset(dataset, idx_val)) for (idx_train, idx_val) in zip(idxs_train, idxs_val)]