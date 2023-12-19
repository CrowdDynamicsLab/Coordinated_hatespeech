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
    def __init__(self, journalist):    
        self.conversatin_id=journalist['conversation_id']
        self.user_id=journalist['user_id']
        self.reference_id=journalist['reference_id']
        self.tweet_id=journalist['tweet_id']
        self.type=journalist['type']
        self.sens=journalist['possibly_sensitive']
        self.lang=journalist['lang']
        self.reply=journalist['reply_settings'] 
        self.created_at=journalist['created_at'] 
        self.retweet_count=journalist['retweet_count']
        self.like_count=journalist['like_count']
        self.quote_count=journalist['quote_count']
        self.impression_count=journalist['impression_count'] 
        self.mentions=journalist['mentions']
        self.urls=journalist['urls']
        self.labels=journalist['labels'] 

        """if delta_times is not None:
            self.in_times = [torch.Tensor(t[:-1]) for t in delta_times]
            self.out_times = [torch.Tensor(t[1:]) for t in delta_times]"""

    def __getitem__(self, key):
        return self.conversatin_id[key], self.reference_id[key], \
                self.tweet_id[key], self.type[key], self.sens[key], self.lang[key], \
                self.reply[key], self.created_at[key], self.retweet_count[key], self.like_count[key], \
                self.quote_count[key], self.impression_count[key], self.mentions[key], self.urls[key], self.labels[key]

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
            "rel": rel
        }