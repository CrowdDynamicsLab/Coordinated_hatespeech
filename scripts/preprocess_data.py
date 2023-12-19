# generate input data
import json
import os
import pandas as pd
from datetime import datetime
import pickle 
from matplotlib import pyplot as plt


# aliceysu
# bainjal
users = []
journalist = 'aliceysu'
data_dir = '../desktop/'
out_dir = '../../data'
path = os.path.join('../desktop/', f'tweets_in_{journalist}_started_convs.json')
column = ['author_id', 'conversation_id', 'user_id', 'reference_id', 'tweet_id', 'type', 'possibly_sensitive', 'lang', 
           'reply_settings', 'created_at', 'retweet_count', 'reply_count', 'like_count', 'quote_count', 
           'impression_count', 'context', 'mentions', 'annotations', 'urls']
# '../desktop/tweets_in_aliceysu_started_convs.json'
with open(path) as f:
    for line in f:
        #print(line)
        users.append(json.loads(line))
    #users = [json.loads(line) for line in f]

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

def generate_csv(user):
    # ignore text
    user_info = []
    count = 0
    for item in user:
        temp = [item['in_reply_to_user_id'], map_id[item['conversation_id']], item['author_id'], 
                map_id[item['referenced_tweets'][0]['id']], map_id[item['edit_history_tweet_ids'][0]],
                map_type[item['referenced_tweets'][0]['type']], int(item['possibly_sensitive'] == True), map_lan[item['lang']], 
                map_reply[item['reply_settings']], item['created_at'], 
                item['public_metrics']['retweet_count'], item['public_metrics']['reply_count'],
                item['public_metrics']['like_count'], item['public_metrics']['quote_count'], 
                item['public_metrics']['impression_count']] #, item['text']

        if 'context_annotations' in item.keys():
            temp_con = []
            for x in item['context_annotations']:
                temp_con.append(x['domain']['name'] + ' ' + x['entity']['name'])
            temp.append(temp_con)
        else: temp.append(0)

        if 'entities' not in item.keys():
            temp.extend([0,0,0])
            continue

        if 'mentions' in item['entities'].keys():
            temp.append([item['entities']['mentions'][i]['id'] for i in range(len(item['entities']['mentions']))])
        else: temp.append(0)

        if 'annotations' in item['entities'].keys():
            temp_anno = []
            for x in item['entities']['annotations']:
                temp_anno.append(x['type'] + ' ' + x['normalized_text'])
            temp.append(temp_anno)
        else: temp.append(0)

        if 'urls' in item['entities'].keys():
            #temp.append(item['entities']['urls'])
            temp.append(1)
        else: temp.append(0)

        user_info.append(temp)

        count += 1
        
    return user_info


map_id = map_ids(users[0])
map_lan = map_lans(users[0])
map_type = map_types(users[0])
map_reply = map_replies(users[0])

user_conv = generate_csv(users[0])
user_conv_info = pd.DataFrame(generate_csv(users[0]), columns = column)
user_data = user_conv_info.to_dict('list')

def save():
    # save files
    with open(os.path.join(out_dir, f'{journalist}_ids.pkl'), 'wb') as f:
        pickle.dump(map_id, f)

    with open(os.path.join(out_dir, f'{journalist}_lan.pkl'), 'wb') as f:
        pickle.dump(map_lan, f)
        
    with open(os.path.join(out_dir, f'{journalist}_type.pkl'), 'wb') as f:
        pickle.dump(map_type, f)
        
    with open(os.path.join(out_dir, f'{journalist}_reply.pkl'), 'wb') as f:
        pickle.dump(map_reply, f)
            
    with open(os.path.join(out_dir, f'{journalist}_dict.pkl'), 'wb') as f:
        pickle.dump(user_conv, f)
            
    user_conv_info.to_csv(os.path.join(out_dir, f'{journalist}_conv.csv'), index=False)

if __name__=='__main__':
    save()