import json
import re
import os
from os.path import join
from torch.utils.data import Dataset


class SampleDataset(Dataset):
    def __init__(self, split : str, path : str) ->None:
        assert split in ['train','valid','test']
        self._data_path=join(path,split)
        self._n_data=_count_data(self._data_path)

    def __len__(self) -> int:
        return self._n_data

    def __getitem__(self, i : int):
        with open(join(self._data_path,'{}.json'.format(i+1))) as f:
            sample=json.load(f)
        return sample



def _count_data(path):

        matcher = re.compile(r'[0-9]+\.json')
        match = lambda name: bool(matcher.match(name))
        names = os.listdir(path)
        n_data = len(list(filter(match, names)))
        return n_data
