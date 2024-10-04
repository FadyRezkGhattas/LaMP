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

        item_bytes = self.reader.get(
            f"{str(index).zfill(10)}".encode("utf-8")  # eg, b'0000000005'
        )
        item = pickle.loads(item_bytes)
        model_vec = item['model_vec'].clone().detach()
 
        return model_vec