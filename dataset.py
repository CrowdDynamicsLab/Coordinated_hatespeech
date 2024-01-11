import numpy as np
import torch
import pickle as pkl

from pathlib import Path
from sklearn.model_selection import train_test_split

from .utils.utils import *
from .utils.tree_utils import *

class Batch():
    def __init__(self, data, labels, prob, global_path, local_path, masks):
        self.data = data
        self.labels = labels
        self.prob = prob
        self.global_path = global_path
        self.local_path = local_path
        self.masks = masks

def collate(batch):
    batch_li = [list(item) for item in batch]
    data_temp = [row[0] for row in batch_li]
    labels_temp = [torch.Tensor(row[1]) for row in batch_li]
    prob_temp = [torch.Tensor(row[2]) for row in batch_li]
    global_path_temp = [row[3] for row in batch_li]
    local_path_temp = [row[4] for row in batch_li]
    

    padded_data, masks = pad_sequences(data_temp, max_dim=2000, pad_token=0)
    #padded_labels = pad_labels(labels_temp, max_dim=2000, pad_token=0)
    #padded_prob, _ = pad_sequences(prob_temp, max_dim=2000, pad_token=0)
    padded_global, _ = pad_sequences(summ(global_path_temp), max_dim=2000, pad_token=0)
    padded_local = pad_matrix(local_path_temp, max_dim=2000, pad_token=0)

    data= torch.tensor(padded_data).to(torch.int64)
    #labels = torch.tensor(padded_labels).to(torch.int64)
    #prob = torch.tensor(padded_prob).to(torch.int64)
    global_path = torch.tensor(padded_global).to(torch.int64)
    local_path = torch.tensor(padded_local).to(torch.int64)
    labels = torch.nn.utils.rnn.pad_sequence(labels_temp, batch_first=True)
    prob = torch.nn.utils.rnn.pad_sequence(prob_temp, batch_first=True)
    #global_path = torch.nn.utils.rnn.pad_sequence(global_path_temp, batch_first=True)
    #local_path = torch.nn.utils.rnn.pad_sequence(local_path_temp, batch_first=True)
    #print(masks)
    
    #out_tweet_type = torch.nn.utils.rnn.pad_sequence(out_tweet_types, batch_first=True)
    #print("start")
    return Batch(data, labels, prob, global_path, local_path, torch.tensor(masks))
    
    
class TreeDataset(torch.utils.data.Dataset):
    """Dataset class containing variable length sequences.

    Args:
        delta_times: Inter-arrival times between events. List of variable length sequences.

    """
    def __init__(self, data, prob, global_path, local_path, label):  
        self.data = data
        self.prob = prob
        self.global_path = global_path
        self.local_path = local_path
        self.labels = label

    def __getitem__(self, key):
        return self.data[key], self.labels[key], \
                self.prob[key], self.global_path[key], self.local_path[key]
    def __len__(self):
        return len(self.labels)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, fps):
        super().__init__()
        self.loaders = {name: JsonLoader(fp) for name, fp in fps.items()}
        assert min(map(len, self.loaders.values())) == max(map(len, self.loaders.values())), \
             list(map(lambda x: (x, len(self.loaders[x])), self.loaders.keys()))
        self.len = len(list(self.loaders.values())[0]) 

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return {name: loader[idx] for name, loader in self.loaders.items()}

    @staticmethod
    def collate(seqs):
        pad_idx = 0
        max_len = max(len(dp[0]) for dp in seqs)
        input_ids, input_values, input_labels = [], [], []
        target_ids, target_values, target_labels = [], [], []
        extended = []
        position_seqs = [] 
        rel_mask = torch.zeros((len(seqs), max_len - 1, max_len - 1)).long() 

        # here
        code_td_paths_rep = torch.zeros(len(seqs), max_len - 1, td_max_depth, 2, dtype=torch.int)
        code_local_relations_rep = torch.zeros(len(seqs), max_len - 1, max_len - 1, 3, dtype=torch.long)

        for i, dp in enumerate(seqs):
            tweet_id, x, pos, l = dp[0], dp[1], dp[2], dp[3]
            
            padding = [[pad_idx] * len(x[0])] * (max_len - len(l))
            input_values.append(x[:-1] + padding)
            target_values.append(x[1:] + padding)
            
        
            
        return {
            "ids": {
                "tweet_id": tweet_id,
            },
            "input_seq": {
                "values": torch.tensor(input_values), 
                #"labels": torch.tensor(input_labels)
            },
            "target_seq": {
                "values": torch.tensor(target_values)
                #"labels": torch.tensor(target_labels)
            },
            "positions": pos,
            "rel": rel,
            "prob": prob
        }