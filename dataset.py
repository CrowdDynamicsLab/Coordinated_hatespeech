import numpy as np
import torch
import pickle as pkl

from pathlib import Path
from sklearn.model_selection import train_test_split

class Batch():
    def __init__(self, conv_id, ref_id, tweet_id, type, sens, lang, reply, retweet_count, like_count, quote_count, 
                 impression_count, mentions, urls, labels, length):
        self.conv_id=conv_id
        self.ref_id=ref_id
        self.tweet_id=tweet_id
        self.type=type
        self.sens=sens
        self.lang=lang
        self.reply=reply
        self.retweet_count=retweet_count
        self.like_count=like_count
        self.quote_count=quote_count
        self.impression_count=impression_count
        self.mentions=mentions
        self.urls=urls
        self.labels=labels
        self.length=length

def collate(batch):
    conv_id = torch.tensor([item[0] for item in batch]).to(torch.int64)
    ref_id = torch.tensor([item[1] for item in batch]).to(torch.int64)
    tweet_id = torch.tensor([item[2] for item in batch]).to(torch.int64)
    type = torch.tensor([item[3] for item in batch]).to(torch.int64)
    sens = torch.tensor([item[4] for item in batch]).to(torch.int64)
    lang = torch.tensor([item[5] for item in batch]).to(torch.int64)
    reply = torch.tensor([item[6] for item in batch]).to(torch.int64)
    #created_at = torch.tensor([item[7] for item in batch])
    retweet_count = torch.tensor([item[8] for item in batch]).to(torch.int64)
    like_count = torch.tensor([item[9] for item in batch]).to(torch.int64)
    quote_count = torch.tensor([item[10] for item in batch]).to(torch.int64)
    impression_count = torch.tensor([item[11] for item in batch]).to(torch.int64)
    mentions = torch.tensor([item[12] for item in batch]).to(torch.int64)
    urls = torch.tensor([item[13] for item in batch]).to(torch.int64)
    labels = torch.tensor([item[14] for item in batch]).to(torch.int64)
    length = torch.Tensor([len(item) for item in batch]).to(torch.int64)
    
    #out_tweet_type = torch.nn.utils.rnn.pad_sequence(out_tweet_types, batch_first=True)
    #print("start")
    #return Batch(in_time, out_time, length, in_mark=in_mark, out_mark=out_mark, in_tweet_type=in_tweet_type, out_tweet_type=out_tweet_type, index=index)
    #return Batch(conv_id, ref_id, tweet_id, type, sens, lang, reply, retweet_count, like_count, quote_count, 
    #             impression_count, mentions, urls, labels, length)
    return {
            "ids": {
                "conv_id": conv_id,
                "ref_id": ref_id, 
                "tweet_id": tweet_id
            },
            "features": features,
            "positions": positions,
            "labels": labels
        }
    
    
class TreeDataset(torch.utils.data.Dataset):
    """Dataset class containing variable length sequences.

    Args:
        delta_times: Inter-arrival times between events. List of variable length sequences.

    """
    def __init__(self, journalist, prob, global_path, local_path):  
        self.id = journalist[['tweet_id']].to_numpy()
        self.user = journalist[['user_id']].to_numpy()
        self.feature = journalist[['type', 'possibly_sensitive', 'lang', 'reply_settings', 
                                   'retweet_count', 'reply_count', 'like_count', 'quote_count',
                                    'impression_count', 'mentions', 'urls']].to_numpy()

        self.label = journalist[['labels']].to_numpy()
        self.prob = prob
        self.global_path = global_path
        self.local_path = local_path

    def __getitem__(self, key):
        return self.feature[key], self.label[key], \
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