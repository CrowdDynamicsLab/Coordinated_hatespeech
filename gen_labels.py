import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import json
from datetime import datetime

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torch.optim import Adam
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, BertModel

journalist = 'sallykohn'
data_dir = '../data'

def map_ids(u):
    map_id = {}
    count = 0
    for i in range(len(u)):
        if u[i]['conversation_id'] not in map_id.keys():
            map_id[u[i]['conversation_id']] = count
            count += 1
        if u[i]['referenced_tweets'][0]['id'] not in map_id.keys():
            map_id[u[i]['referenced_tweets'][0]['id']] = count
            count += 1
            
        if u[i]['edit_history_tweet_ids'][0] not in map_id.keys():
            map_id[u[i]['edit_history_tweet_ids'][0]] = count
            count += 1
    return map_id

def map_lans(u):
    map_lan = {}
    count = 0
    for i in range(len(u)):
        if u[i]['lang'] not in map_lan.keys():
            map_lan[u[i]['lang']] = count
            count += 1
    return map_lan

def map_types(u):
    map_type = {}
    count = 0
    for i in range(len(u)):
        if u[i]['referenced_tweets'][0]['type'] not in map_type.keys():
            map_type[u[i]['referenced_tweets'][0]['type']] = count
            count += 1
    return map_type

def map_replies(u):
    map_reply = {}
    count = 0
    for i in range(len(u)):
        if u[i]['reply_settings'] not in map_reply.keys():
            map_reply[u[i]['reply_settings']] = count
            count += 1
    return map_reply

def generate_csv(user, label):
    user_info = []
    count = 0
    for item in user:
        temp = [item['in_reply_to_user_id'], map_id[item['conversation_id']], item['author_id'], 
                map_id[item['referenced_tweets'][0]['id']], map_id[item['edit_history_tweet_ids'][0]],
                map_type[item['referenced_tweets'][0]['type']], int(item['possibly_sensitive'] == True), map_lan[item['lang']], 
                map_reply[item['reply_settings']], item['created_at'], 
                item['public_metrics']['retweet_count'], item['public_metrics']['reply_count'],
                item['public_metrics']['like_count'], item['public_metrics']['quote_count'], 
                item['public_metrics']['impression_count'], item['text']]

        if count < len(label):
            temp.append(label[count])
        else:
            temp.append(np.nan)

        if 'context_annotations' in item.keys():
            # temp_con = []
            # for x in item['context_annotations']:
            #     temp_con.append(x['domain']['name'] + ' ' + x['entity']['name'])
            # temp.append(temp_con)
            temp.append(item['context_annotations'])
        else: temp.append(0)

        if 'entities' not in item.keys():
            temp.extend([0,0,0])
            continue
        """if 'entities' not in item.keys():
            temp.append([0, 0])
            continue"""
        temp_men = []
        #print(count, item)
        if 'mentions' in item['entities'].keys():
            temp_men.append([item['entities']['mentions'][i]['id'] for i in range(len(item['entities']['mentions']))])
            temp.append(len(temp_men))
        else: temp.append(0)

        if 'annotations' in item['entities'].keys():
            # temp_anno = []
            # for x in item['entities']['annotations']:
            #     temp_anno.append(x['type'] + ' ' + x['normalized_text'])
            temp.append(item['entities']['annotations'])
        else: temp.append(0)

        if 'urls' in item['entities'].keys():
            #temp.append(item['entities']['urls'])
            temp.append(1)
        else: temp.append(0)
        
        user_info.append(temp)

        count += 1
        
    return user_info, len(label)

# Define Dataset
class TwitterDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        # Convert labels to tensor and adjust for 1, -1, 0
        label = int(self.labels[item])
        label = 2 if label == -1 else label  # Map -1 to 2

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }
    
# Tokenizer
def BertModel(labeled_df, unlabeled_df):
    # Tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    max_len = 200

    # Create dataset
    labeled_dataset = TwitterDataset(labeled_df['text'], labeled_df['labels'], tokenizer, max_len)

    # DataLoader
    data_loader = DataLoader(labeled_dataset, batch_size=8)

    # Model (specify num_labels=3 for three class classification)
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Training Loop
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    epochs = 5

    for epoch in range(epochs):
        model.train()
        for batch in tqdm(data_loader):
            #print(batch['text'])
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            #print(input_ids, attention_mask, labels)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch} completed')

    # Inference on Unlabeled Data
    model.eval()
    predictions = []

    for text in labeled_df['text']:
        encoding = tokenizer.encode_plus(
            text, 
            add_special_tokens=True, 
            max_length=max_len, 
            return_attention_mask=True, 
            padding='max_length', 
            truncation=True, 
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            prediction = torch.argmax(outputs.logits, dim=1)
            predictions.append(prediction.cpu().numpy())
    labeled_df['predicted_label'] = predictions
    """truth = []
    for l in labeled_df['labels']:
        if l != -1:
            truth.append(int(l))
        else:
            truth.append(2)"""
    acc = sum(labeled_df['labels'] == labeled_df['predicted_label']) / len(labeled_df)
    print("acc", acc)

    predictions_un = []
    for text in tqdm(unlabeled_df['text']):
        encoding = tokenizer.encode_plus(
            text, 
            add_special_tokens=True, 
            max_length=max_len, 
            return_attention_mask=True, 
            padding='max_length', 
            truncation=True, 
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            prediction = torch.argmax(outputs.logits, dim=1)
            predictions_un.append(prediction.cpu().numpy())

    unlabeled_df['predicted_label'] = predictions_un
    return labeled_df, unlabeled_df
    #labeled_df.to_csv(os.path.join(data_dir, f'{journalist}/{journalist}_conv_labeled.csv'), index=False) 
    #unlabeled_df.to_csv(os.path.join(data_dir, f'{journalist}/{journalist}_conv_unlabeled.csv'), index=False)

if __name__=='__main__':
    path = os.path.join(data_dir, f'{journalist}/tweets_in_{journalist}_started_convs.json')

    column = ['author_id', 'conversation_id', 'user_id', 'reference_id', 'tweet_id', 'type', 'possibly_sensitive', 'lang', 
           'reply_settings', 'created_at', 'retweet_count', 'reply_count', 'like_count', 'quote_count', 
           'impression_count', 'text', 'labels', 'context', 'mentions', 'annotations', 'urls']
    column_label = ['author_id', 'conversation_id', 'user_id', 'reference_id', 'tweet_id', 'type', 'possibly_sensitive', 'lang', 
            'reply_settings', 'created_at', 'retweet_count', 'reply_count', 'like_count', 'quote_count', 
            'impression_count', 'mentions', 'urls', 'context', 'annotations', 'labels']
    column_text = ['author_id', 'conversation_id', 'user_id', 'reference_id', 'tweet_id', 'text', 'labels']

    users = []
    with open(path) as f:
        for line in f:
            users.append(json.loads(line))
            
    df = pd.read_csv(os.path.join(data_dir, f'{journalist}/{journalist}_context.csv'))
    lab = list(df['labels'])
    
    count = 0
    for i in range(len(lab)):
        #if lab[i] == '-1' or lab[i] == '0' or lab[i] == '1' or lab[i] == -1 or lab[i] == 1 or lab[i] ==0:
        if lab[i] == 0 or lab[i] == 1 or lab[i] ==2:
            count += 1
        else: break

    label_real = lab[:count]
    if type(label_real[0]) == str:
        print("yes")
        labels = {'-1': 0, '0': 1, '1': 2, np.nan: 9000}
    else:
        labels = {0.0: 0, 1.0: 1, 2.0: 2, np.nan: 9000}
    label = [labels[l] for l in label_real]

    print(torch.cuda.is_available())

    map_id = map_ids(users[0])
    map_lan = map_lans(users[0])
    map_type = map_types(users[0])
    map_reply = map_replies(users[0])

    user_conv, num_label = generate_csv(users[0], label)
    user_conv_old = pd.DataFrame(user_conv, columns = column)

    user_text = user_conv_old[column_text]
    user_conv_info = user_conv_old[column_label]

    # Split the data into labeled and unlabeled
    labeled_text = user_text.dropna(subset=['labels']).reset_index(drop=True)  # Assuming rows with missing labels are NaN
    unlabeled_text = user_text[user_text['labels'].isna()].reset_index(drop=True)

    
    #labeled_df = labeled_text.reset_index(drop=True)
    #unlabeled_df = unlabeled_df.reset_index(drop=True)
    labeled, unlabeled = BertModel(labeled_text, unlabeled_text)

    labeled.labels = [label for label in labeled['labels']]
    unlabeled.labels = [label[0] for label in unlabeled['predicted_label']]

    labeled_text =pd.concat([labeled, unlabeled])
    labeled_list = list(labeled_text.labels)
    user_conv_info['labels'] = labeled_list


    date_format = "%Y-%m-%dT%H:%M:%S.%fZ" 
    for i in range(len(user_conv_info['created_at'])):
        user_conv_info['created_at'][i] = datetime.strptime(user_conv_info['created_at'][i], date_format)
    user_conv_sort = user_conv_info.sort_values(by=['created_at'], ascending=True)

    out_path = os.path.join(data_dir, f'{journalist}/{journalist}_conv_labels.csv')
    user_conv_sort.to_csv(out_path, index=False)