""" CNN/DM dataset"""
import json
import re
import os
from torch.utils.data import Dataset

pjoin = os.path.join

class Dataset(Dataset):
    def __init__(self, dataset, data_dir, split):
        self.dataset = dataset
        self.split = split
        self.data_dir = data_dir
        self.data_size = len(os.listdir(data_dir))

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        with open(pjoin(self.data_dir, '%d.json'%idx)) as f:
            data = json.loads(f.read())
        return data


def _count_data(path):
    """ count number of data in the given path"""
    matcher = re.compile(r'[0-9]+\.json')
    match = lambda name: bool(matcher.match(name))
    names = os.listdir(path)
    n_data = len(list(filter(match, names)))
    return n_data
