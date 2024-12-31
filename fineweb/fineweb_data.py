import torch
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
import os

import sys, os; sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")
from utils.utils import format_large_number

def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32) # added after video
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

# NOTE: this Dataset class will cache the shard data for the current index, which means that a new shard will only be loaded once the shard is exhausted
# This process becomes very slow if we set shuffle=True in a dataloader because it will likely need to load a different shard for each sequence in each batch

class FinewebDataset(Dataset):
    def __init__(self, split, sequence_length, data_root='data/edu_fineweb10B', load_shard_function=load_tokens, transform=None):
        super().__init__()
        self.split = split
        self.data_root = data_root
        self.load_function = load_shard_function
        self.transform = transform
        self.sequence_length = sequence_length

        # get the shard filenames
        shard_paths = os.listdir(data_root)
        shard_paths = [s for s in shard_paths if split in s]
        shard_paths = sorted(shard_paths)
        shard_paths = [os.path.join(data_root, s) for s in shard_paths]

        self.shard_paths = shard_paths

        self.shard_index_start = []  # Stores the offsets of each shard's data in the concatenated dataset
        self.shard_lengths = [] # number of sequences in each shard
        self.num_sequences = 0

        shard_iter = tqdm(self.shard_paths, desc='Processing Shards...')
        for shard_path in shard_iter:
            shard_n_sequences = len(self.reshape_shard(
                self.load_function(shard_path),
                sequence_length=self.sequence_length+1))

            self.shard_lengths.append(shard_n_sequences)

            # start index for shard is the total number of sequences so far
            self.shard_index_start.append(self.num_sequences)

            self.num_sequences += shard_n_sequences

            shard_iter.set_postfix(num_sequences=self.num_sequences, num_tokens=self.num_sequences * (self.sequence_length + 1))

        self.num_tokens = self.num_sequences * (self.sequence_length + 1)
        print('# of Tokens:', format_large_number(self.num_tokens))
        print('# of Sequences:', format_large_number(self.num_sequences))
        print('Sequence Length:', self.sequence_length)

        self.current_shard_index = None
        self.current_shard_data = None

    def reshape_shard(self, shard, sequence_length):
        """reshapes shard from (n_tokens,) to (n_sequences, sequence_length), discarding any remaining tokens"""

        n_sequences = len(shard) // sequence_length
        shard = shard[:n_sequences * sequence_length]
        shard = shard.reshape(n_sequences, sequence_length)
        return shard

    def _get_shard_index_and_offset(self, index):
        """Finds the shard and the relative offset for a given index."""
        for i, offset in enumerate(self.shard_index_start):
            if index < offset + self.shard_lengths[i]: # index is within the i-th shard
                shard_index = i
                relative_offset = index - offset
                return shard_index, relative_offset
        raise IndexError(f"Index {index} is out of bounds for dataset of size {self.num_sequences}")

    def _load_shard(self, shard_index):
        """Loads the shard data if it is not already loaded."""
        if self.current_shard_index != shard_index:
            self.current_shard_index = shard_index

            self.current_shard_data = self.reshape_shard(
                self.load_function(self.shard_paths[shard_index]),
                sequence_length=self.sequence_length+1) # +1 for target

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, index):
        if index < 0 or index >= self.num_sequences:
            raise IndexError(f"Index {index} is out of bounds for dataset of size {self.num_sequences}")

        shard_index, relative_offset = self._get_shard_index_and_offset(index)
        self._load_shard(shard_index)

        data_point = self.current_shard_data[relative_offset]

        if self.transform:
            data_point = self.transform(data_point)

        return data_point[:-1], data_point[1:]
