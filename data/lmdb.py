import pickle
from tensorfn.data import LMDBReader
import torch
from torch.utils.data import Dataset

class LMDBDataset(Dataset):

    def __init__(self, path):
        
        self.reader = LMDBReader(path, reader="raw")

    def __len__(self):
        return len(self.reader)
    
    def __getitem__(self, index):
        return self._read_element(index)
    
    def get_data_stats(self):
        vecs = []
        for i in range(len(self.reader)):
            vecs.append(self._read_element(i))
        vecs = torch.stack(vecs)
        means = torch.mean(vecs, dim=0)
        std = torch.std(vecs, dim=0)
        return means, std

    def _read_element(self, index):
        item_bytes = self.reader.get(
            f"{str(index).zfill(10)}".encode("utf-8")  # eg, b'0000000005'
        )
        item = pickle.loads(item_bytes)
        return torch.tensor(item['model_vec']).float()