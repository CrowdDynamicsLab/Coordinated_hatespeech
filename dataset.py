import numpy as np
import torch 
import pickle as pkl

from pathlib import Path
from sklearn.model_selection import train_test_split

from utils.utils import *
from utils.tree_utils import *

class Batch():
    def __init__(self, idx, uid, data, labels, prob, global_path, local_path, masks):
        self.id = idx
        self.uid = uid
        self.data = data
        self.labels = labels
        self.prob = prob
        self.global_path = global_path
        self.local_path = local_path
        self.masks = masks

"""def collate(batch):
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
    global_path = torch.tensor(np.array(padded_global)).to(torch.int64)
    #print(type(padded_local), type(padded_local[0]))
    local_path = torch.tensor(np.array(padded_local)).to(torch.int64)
    labels = torch.nn.utils.rnn.pad_sequence(labels_temp, batch_first=True)
    prob = torch.nn.utils.rnn.pad_sequence(prob_temp, batch_first=True)
    #global_path = torch.nn.utils.rnn.pad_sequence(global_path_temp, batch_first=True)
    #local_path = torch.nn.utils.rnn.pad_sequence(local_path_temp, batch_first=True)
    #print(masks)
    
    #out_tweet_type = torch.nn.utils.rnn.pad_sequence(out_tweet_types, batch_first=True)
    #print("start")
    return Batch(data, labels, prob, global_path, local_path, torch.tensor(masks))
"""    
def collate(batch):
    batch_li = [list(item) for item in batch]
    data_temp = [row[0] for row in batch_li]
    #print("?????", [row[1] for row in batch_li], type(batch_li[0][1]))
    labels_temp = [torch.Tensor(row[1]) for row in batch_li]
    prob_temp = [torch.Tensor(row[2]) for row in batch_li]
    global_path_temp = [row[3] for row in batch_li]
    local_path_temp = [row[4] for row in batch_li]
    idx_temp = [torch.Tensor(row[5]) for row in batch_li]
    uid_temp = [torch.Tensor(row[6]) for row in batch_li]
    

    padded_data, masks = pad_sequences(data_temp, max_dim=500, pad_token=0)
    padded_labels = pad_labels(labels_temp, max_dim=500, pad_token=0)
    padded_idx = pad_labels(idx_temp, max_dim=500, pad_token=0)
    padded_uid = pad_labels(uid_temp, max_dim=500, pad_token=0)
    padded_prob, _ = pad_sequences(prob_temp, max_dim=500, pad_token=0)
    padded_global, _ = pad_sequences(summ(global_path_temp), max_dim=500, pad_token=0)
    padded_local = pad_matrix(local_path_temp, max_dim=500, pad_token=0)

    data= torch.tensor(padded_data).to(torch.int64)
    labels = torch.tensor(padded_labels).to(torch.int64)
    idx = torch.tensor(padded_idx).to(torch.int64)
    uid = torch.tensor(padded_uid).to(torch.int64)
    prob = torch.tensor(padded_prob).to(torch.float64)
    global_path = torch.tensor(np.array(padded_global)).to(torch.int64)
    #print(type(padded_data), type(padded_data[0]))
    local_path = torch.tensor(np.array(padded_local)).to(torch.int64)
    #labels = torch.nn.utils.rnn.pad_sequence(labels_temp, batch_first=True)
    #prob = torch.nn.utils.rnn.pad_sequence(prob_temp, batch_first=True)
    #global_path = torch.nn.utils.rnn.pad_sequence(global_path_temp, batch_first=True)
    #local_path = torch.nn.utils.rnn.pad_sequence(local_path_temp, batch_first=True)
    #print(masks)
    
    #out_tweet_type = torch.nn.utils.rnn.pad_sequence(out_tweet_types, batch_first=True)
    #print("start")
    return Batch(idx, uid, data, labels, prob, global_path, local_path, torch.tensor(masks))
    
   
class TreeDataset(torch.utils.data.Dataset):
    """Dataset class containing variable length sequences.

    Args:
        delta_times: Inter-arrival times between events. List of variable length sequences.

    """
    def __init__(self, ids, uid, data, prob, global_path, local_path, label):  
        self.id = ids
        self.uid = uid
        self.data = data
        self.prob = prob
        self.global_path = global_path
        self.local_path = local_path
        self.labels = label

    def __getitem__(self, key):
        return self.data[key], self.labels[key], \
                self.prob[key], self.global_path[key], self.local_path[key], self.id[key], self.uid[key]
    def __len__(self):
        return len(self.labels)


