import math
import pickle
import numpy as np
import pandas as pd
import pickle as pkl
from scipy.stats import beta
from tqdm import tqdm
from datetime import datetime
from collections import defaultdict

from transformers import BertTokenizer, BertModel

import os
import ast
#from Content import *
#from Venue import *

import torch

def get_context(conv):
    contexts = []
    for item in conv['context']:
        temp_s = ''
        if item == '0':
            contexts.append('Unified Twitter Taxonomy')

        else:
            for i in ast.literal_eval(item):
                temp_s += (i['domain']['name'] + ' ' + i['entity']['name'])
            contexts.append(temp_s)
    return contexts

def get_anno(conv):
    annotations = []
    for item in conv['annotations']:
        temp_s = ''
        if item == '0':
            annotations.append('Others')

        else:
            for i in ast.literal_eval(item):
                temp_s += i['type'] + ' ' + i['normalized_text']

            annotations.append(temp_s)
    return annotations

def cosine_similarity(vec1, vec2):
    #print(vec.size(), vec2.size())
    dot_product = torch.dot(vec1.view(-1), vec2.view(-1))
    norm_vec1 = torch.norm(vec1)
    norm_vec2 = torch.norm(vec2)
    similarity = dot_product / (norm_vec1 * norm_vec2)
    return similarity

def context_sim(df):
    context_sims = []

    # Initialize tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    for index, item in tqdm(df.iterrows()):
        # Tokenize input text and convert to tensor
        c1 = item['topics'][0]
        c2 = item['topics'][1]
        
        a1 = item['ppls'][0]
        a2 = item['ppls'][1]
        inputs1 = tokenizer(c1, return_tensors="pt").to(device)
        inputs2 = tokenizer(c2, return_tensors="pt").to(device)

        # Generate embeddings
        with torch.no_grad():
            outputs1 = model(**inputs1)
            outputs2 = model(**inputs2)

        # The last hidden state is the sequence of hidden states of the last layer of the model
        last_hidden_states1 = outputs1.last_hidden_state
        last_hidden_states2 = outputs2.last_hidden_state

        # Optionally, use the [CLS] token's embedding as the representation for the entire sentence
        sentence_embedding1 = last_hidden_states1[:, 0, :]
        sentence_embedding2 = last_hidden_states2[:, 0, :]

        cosine = cosine_similarity(sentence_embedding1,sentence_embedding2)
        context_sims.append(cosine.cpu())
    
    return context_sims

def anno_sim(df):
    # Initialize tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    anno_sims = []
    for index, item in tqdm(df.iterrows()):
        # Tokenize input text and convert to tensor
        c1 = item['topics'][0]
        c2 = item['topics'][1]
        
        a1 = item['ppls'][0]
        a2 = item['ppls'][1]
        inputs1 = tokenizer(a1, return_tensors="pt").to(device)
        inputs2 = tokenizer(a2, return_tensors="pt").to(device)

        # Generate embeddings
        with torch.no_grad():
            outputs1 = model(**inputs1)
            outputs2 = model(**inputs2)

        # The last hidden state is the sequence of hidden states of the last layer of the model
        last_hidden_states1 = outputs1.last_hidden_state
        last_hidden_states2 = outputs2.last_hidden_state

        # Optionally, use the [CLS] token's embedding as the representation for the entire sentence
        sentence_embedding1 = last_hidden_states1[:, 0, :]
        sentence_embedding2 = last_hidden_states2[:, 0, :]

        cosine = cosine_similarity(sentence_embedding1,sentence_embedding2)
        anno_sims.append(cosine.cpu())
    
    return anno_sims

def get_time(df):
    anchor_time = []
    # date_format = "%Y-%m-%dT%H:%M:%S" 
    date_format = '%Y-%m-%d %H:%M:%S'
    conv_ids = list(set(df['conversation_id']))
    max_len = 0
    max_id = 0
    for item in conv_ids:
        if len(df[df['conversation_id']==item]) > max_len:
            max_len = len(df[df['conversation_id']==item])
            max_id = item

    test_s = df[df['conversation_id']==max_id]
    for index, item in test_s.iterrows():
        anchor_time.append(datetime.strptime(item['created_at'][:19], date_format))
    anchor = anchor_time[1] - anchor_time[0]
    anchor_hours = divmod(anchor.total_seconds() , 3600)[0]  

    ref_ids = []
    for conv in conv_ids:
        test_d = df[df['conversation_id'] == conv]
        time0 = datetime.strptime(test_d.iloc[0]['created_at'][:19], date_format)
        k = 0
        for index, item in test_d.iterrows():
            tweet_id = item['tweet_id']
            ref_id = item['reference_id']
            conv_id = item['conversation_id']
            time1 = datetime.strptime(item['created_at'][:19], date_format)
            if len(test_d) == 1:
                journal_sort.at[index, 'time gap']= float(anchor_hours) / 120
                continue
            if ref_id not in ref_ids:
                if tweet_id == test_d.iloc[0]['tweet_id']:
                    time2 = datetime.strptime(item['created_at'][:19], date_format) - anchor
                else:
                    time2 = datetime.strptime(test_d.iloc[k-1]['created_at'][:19], date_format)
            else:
                #print(test_d[test_d['tweet_id']==ref_id]['created_at'][:19].item())
                time2 = datetime.strptime(test_d[test_d['tweet_id']==ref_id]['created_at'].item()[:19], date_format)

            ref_ids.append(tweet_id)
            gap_in_s = (time1 - time2).total_seconds() 

            df.at[index, 'time gap']= float(divmod(gap_in_s, 3600)[0]) / 120

            k += 1
    return df



def cal_cite_edgeprobs(df):
    superbeta = beta(a=10,b=1)
    edge_prob = []
    for index, item in df.iterrows():
         
        ppl_sim = float(item['ppls_sim'].item())
        ppl_dif = 1 - ppl_sim

        topic_sim = float(item['topics_sim'].item())
        topic_dif = 1 - topic_sim

        pnormal_latest = superbeta.pdf(1-item['time gap']) + 1e-20
        puniform_latest = 1 / 168

        temp1 = np.array([ppl_sim, ppl_dif])
        temp2 = np.outer([topic_sim,topic_dif],[pnormal_latest,puniform_latest]).flatten()
        result = np.outer(temp1,temp2).flatten()
        edge_prob.append(result)
    
    return np.array(edge_prob, dtype=np.float32)

if __name__=='__main__':
    out_dir = '.'
    journalist = 'sallykohn'
    conv = pd.read_csv(os.path.join(out_dir, f'{journalist}/{journalist}_conv_labels.csv'))

    journal = conv
    contexts = get_context(conv)
    annotations = get_anno(conv)

    journal['context'] = contexts
    journal['annotations'] = annotations
    journal_sort = journal.sort_values(by=['created_at'])

    ref_ids = []

    journal_sort['topics'] = None
    journal_sort['ppls'] = None
    journal_sort['time gap'] = None
    conv_ids = set(journal_sort['conversation_id'])

    for conv in list(conv_ids):
        test_d = journal_sort[journal_sort['conversation_id'] == conv]
        for index, item in test_d.iterrows():
            i = 0
            tweet_id = item['tweet_id']
            ref_id = item['reference_id']
            conv_id = item['conversation_id']
            topic1 = item['context']
            anno1 = item['annotations']
            if len(test_d) == 1:
                topic2 = 'Unified Twitter Taxonomy'
                anno2 = 'Others'
                journal_sort.at[index, 'topics']=[topic1, text2]
                journal_sort.at[index, 'ppls']=[anno1, anno2]
                continue
            if ref_id not in ref_ids:
                if tweet_id == test_d.iloc[0]['tweet_id']:
                    text2 = test_d[test_d['tweet_id']==test_d.iloc[1]['tweet_id']]['context'].item()
                    anno2 = test_d[test_d['tweet_id']==test_d.iloc[1]['tweet_id']]['annotations'].item()
                else:
                    text2 = test_d.iloc[i-1]['context']
                    anno2 = test_d.iloc[i-1]['annotations']
            else:
                #print(tweet_id,ref_id)
                text2 = test_d[test_d['tweet_id']==ref_id]['context'].item()
                anno2 = test_d[test_d['tweet_id']==ref_id]['annotations'].item()

            ref_ids.append(tweet_id)
            
            journal_sort.at[index, 'topics']=[topic1, text2]
            journal_sort.at[index, 'ppls']=[anno1, anno2]
            #print(text2, journal_sort.at[index, 'topics'])
            i += 1
            
    context_sims = context_sim(journal_sort)
    print("context finished")
    anno_sims = anno_sim(journal_sort)
    print("annotations finished")

    journal_sort['topics_sim'] = context_sims
    journal_sort['ppls_sim'] = anno_sims

    journal_sort = get_time(journal_sort)

    print(journal_sort.head())
    edge_prob = cal_cite_edgeprobs(journal_sort)

    journal_sort['sim'] = edge_prob.tolist()
    edgeprob = []
    ids = list(set(journal_sort['conversation_id']))
    for idx in ids:
        df = journal_sort[journal_sort['conversation_id']==idx]
        edgeprob.append(list(df['sim']))

    with open(os.path.join(out_dir, f'{journalist}/{journalist}_edgeprob.pkl'),'wb') as f:
        pickle.dump(edgeprob, f)