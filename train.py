from dataset import *
import os, sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.distributions as td
from copy import deepcopy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import logging
# torch.set_default_tensor_type(torch.cuda.FloatTensor)
import argparse
import sys
import pickle as pkl
import sklearn
from tqdm import tqdm
import json

from sklearn.mixture import GaussianMixture
from collections import Counter
from sklearn.cluster import KMeans
from datetime import datetime

from utils.utils import *
from utils.tree_utils import *


def load_data(args):
    ### Data (normalize input inter-event times, then padding to create dataloaders)
    num_classes, num_sequences = 0, 0
    seq_dataset = []
    arr = []
    dp = []
    rel = []
    
    split = [64, 128]
    val = 0
    journal_sort = pd.read_csv((os.path.join(args.data_dir, f'{args.journalist}_context.csv')))
    ids = list(set(journal_sort['conversation_id']))
    id_pair = {}
    id_conv = {}
    for idx in ids:
        id_pair[idx], id_conv[idx] = create_conversation_list(journal_sort[journal_sort['conversation_id']==idx], idx)
    id_data, data, label = create_data(journal_sort, ids)
    prob = pkl.load(open(os.path.join(args.data_dir, f'{args.journalist}_edgeprob.pkl'), 'rb'))
    
    with open(os.path.join(args.out_fp, f'{args.journalist}_global_path.txt'), "r") as f:
        for line in tqdm(f, total=get_number_of_lines(f)):
            dp.append(json.loads(line.strip()))

    with open(os.path.join(args.out_fp, f'{args.journalist}_local_path.txt'), "r") as f:
        for line in tqdm(f, total=get_number_of_lines(f)):
            rel.append(json.loads(line.strip()))
    
    global_input = convert_global(dp, id_data)
    local_data = convert_local(rel)
    local_mat = generate_local_mat(local_data, id_data)
    local_input = create_mat(local_mat, mat_type='concat')
    logging.info(f'loaded split {args.journalist}...')
    # data - dict: dim_process, devtest, args, train, dev, test, index (train/dev/test given as)
    # data[split] - list dicts {'time_since_start': at, 'time_since_last_event': dt, 'type_event': mark} or
    # data[split] - dict {'arrival_times', 'delta_times', 'marks'}
    # data['dim_process'] = Number of accounts = 119,298
    # num_sequences: number of conversations of a journalist
    num_classes = args.classes
    #num_sequences += len(data[split]['arrival_times'])
    num_sequences = len(set(journal_sort['conversation_id']))
    
    X_train, X_dev, X_test = data[:split[0]], data[split[0]:split[1]], data[split[1]:]
    prob_train, prob_dev, prob_test = prob[:split[0]], prob[split[0]:split[1]], prob[split[1]:]
    global_train, global_dev, global_test = global_input[:split[0]], global_input[split[0]:split[1]], global_input[split[1]:]
    local_train, local_dev, local_test = local_input[:split[0]], local_input[split[0]:split[1]], local_input[split[1]:]
    label_train, label_dev, label_test = label[:split[0]], label[split[0]:split[1]], label[split[1]:]

    d_train = TreeDataset(X_train, prob_train, global_train, local_train, label_train)
    d_val = TreeDataset(X_dev, prob_dev, global_dev, local_dev, label_dev)  
    d_test  = TreeDataset(X_test, prob_test, global_test, local_test, label_test)   

    # for padding input sequences to maxlen of batch for running on gpu, and arranging them by length efficient
    collate = collate  
    dl_train = torch.utils.data.DataLoader(d_train, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    dl_val = torch.utils.data.DataLoader(d_val, batch_size=args.val_batch_size, shuffle=False, collate_fn=collate)
    dl_test = torch.utils.data.DataLoader(d_test, batch_size=args.val_batch_size, shuffle=False, collate_fn=collate)
    return dl_train, dl_val, dl_test, num_classes, num_sequences











if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Tempt model')
    
    
    ## dataset and output directories
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--out_dir', type=str, default='./data/result')
    parser.add_argument('--log_filename', type=str, default='run.log')
    parser.add_argument('--journalist', type=str, default='aliceysu')
    parser.add_argument('--classes', type=int, default=3)
    
    
    ## training arguments
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(args.device)
    if not os.path.isdir(args.out_dir): os.makedirs(args.out_dir)
    np.random.seed(args.seed); torch.manual_seed(args.seed);
    
    logging.basicConfig(
        level=logging.INFO,
        format='[{%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(filename=os.path.join(args.out_dir, args.log_filename)),
            logging.StreamHandler(sys.stdout)
        ]
    ) # logger = logging.getLogger('')
    logging.info('Logging any runs of this program - appended to same file.')
    logging.info('Arguments = {}'.format(args))
    dl_train, dl_val, dl_test, mean_out_train, std_out_train, num_classes, num_sequences = load_data(args)
    print("TRAIN", dl_train)
    logging.info('loaded the dataset and formed torch dataloaders.')

    
    # model, opt = create_model(num_classes, num_sequences, args, mean_out_train, std_out_train)
    # logging.info('model created from config hyperparameters.')
    # gmm = GaussianLaplaceTiedMixture(args.gmm_k, 0, args.mark_embedding_size, device = args.device)
    
    # gmm.to(args.device)
    # train(model, opt, dl_train, dl_val, logging, args.use_marks, args.max_epochs, args.patience, 
    #       args.display_step, args.save_freq, args.out_dir, args.device, args, gmm = gmm)

    # def evaluate(model, dl_list, dl_names, use_marks, device):
    #     # Calculate the train/val/test loss, plot training curve
    #     model.eval()
    #     for dl_, name in zip(dl_list, dl_names):
    #         loss_tot, time_nll, marks_nll, marks_acc = get_total_loss(
    #                 dl_, model, args.use_marks, device)
    #         logging.info(f'{name}: {loss_tot:.4f}')
    #         logging.info(f'TimeNLL:{time_nll:.4f} MarksNLL:{marks_nll:.4f} Acc:{marks_acc:.4f}')
    # dl_list = [dl_train, dl_val, dl_test]
    # dl_names = ['Train', 'Val', 'Test']
    # evaluate(model, dl_list, dl_names, args.use_marks, args.device)
    # # model = torch.load(out_dir + 'best_full_model.pt')
    
    # extract_features(model, logging, args, gmm)
    # logging.info('Finished program.')
    