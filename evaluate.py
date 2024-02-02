import json
import os
import pandas as pd
from datetime import datetime
import pickle 
from matplotlib import pyplot as plt
from collections import Counter
import numpy as np
import torch
import logging
from tqdm import tqdm
import ast
import argparse

from utils.tree_utils import *
from utils.utils import *
from dataset import *
from model import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
import math
from train import load_data, create_model

def evaluate(model, strat_model, dl_test, device):
    # Calculate the train/val/test loss, plot training curve
    model.eval()
    strat_model.eval()
    with torch.no_grad():
        test_loss = 0
        test_steps = 0
        predictions = []
        true_labels = []
        strat_li = []
        predicted_li = []
        for item in tqdm(dl_test):  # Assuming 'val' is your validation dataset
            # Forward pass
            data = item.data.float().to(device)
            dp = item.global_path.float().to(device)
            rel = item.local_path.float().to(device)
            targets = item.labels.long().to(device)
            llh = item.prob.float().to(args.device)
            mask = item.masks.float().to(device)

            output, p_output = model(data, dp, rel, mask)
            prob_output = strat_model(output, p_output.detach())
            strat_li.append(prob_output.tolist())
            _, predicted = torch.max(output.data, 2)
            predicted_li.append(predicted.tolist())
            predictions.extend(predicted.view(-1).tolist())
            true_labels.extend(targets.view(-1).tolist())

        dot_product = torch.sum(prob_output * llh, dim=-1)
        print(f'Test Accuracy: {dot_product}')
        #logging.info(f'{dl_name}: {loss_tot:.4f}')
    return predicted_li, strat_li, true_labels





if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Tempt model')
    
    
    ## dataset and output directories
    parser.add_argument('--data_dir', type=str, default='../data')
    parser.add_argument('--out_dir', type=str, default='../data')
    parser.add_argument('--log_filename', type=str, default='run.log')
    parser.add_argument('--journalist', type=str, default='JiayangFan')
    parser.add_argument('--classes', type=int, default=3)
    parser.add_argument('--best_model_path', type=str, default='best_model.pth')
    parser.add_argument('--best_strat_model_path', type=str, default='best_strat_model.pth')

    ## model encoder parameters
    parser.add_argument('--feature_dim', type=int, default=11)
    parser.add_argument('--global_dim', type=int, default=3)
    parser.add_argument('--embed_size', type=int, default=512)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--forward_expansion', type=int, default=4)
    parser.add_argument('--mode', type=str, default='all')
    parser.add_argument('--depth', type=int, default=1)
    parser.add_argument('--wide', dest='wide', default=True, action='store_true', help='Change back')
    parser.add_argument('--num_layers', type=int, default=3, help='flow-based models.')
    parser.add_argument('--d_ff', type=int, default=None)
    parser.add_argument('--max_len', type=int, default=2000)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--attn', type=str, default='all', help='dp or rel or only')

    
    
    ## training arguments
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--regularization', type=float, default=1e-5)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--max_epochs', type=int, default=10)  # 1000 
    parser.add_argument('--max_loop', type=int, default=1)
    parser.add_argument('--patience', type=int, default=2)
    parser.add_argument('--save_freq', type=int, default=1)
    parser.add_argument('--display_step', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--use_strat', type=bool, default=True)
    
    
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
    logging.info('Arguments = {}'.format(args))
    dl_train, dl_val, dl_test = load_data(args.data_dir, args.journalist, args.classes, args.batch_size, collate)
    logging.info('loaded the dataset and formed torch dataloaders.')
    
    model, opt, strat_model, strat_opt = create_model(args.classes, args)
    logging.info('model created from config hyperparameters.')
    best_model_path = os.path.join(f'{args.journalist}/best_model_w_strat.pth') 
    best_strat_model_path = os.path.join(f'{args.journalist}/best_strat_model_w_strat.pth') 

    model.load_state_dict(torch.load(os.path.join(args.out_dir, best_model_path)))
    strat_model.load_state_dict(torch.load(os.path.join(args.out_dir, best_strat_model_path)))
    

    dl_list = dl_val #[dl_train, dl_val, dl_test]
    dl_names = ['Train', 'Val', 'Test']
    predictions, strat_li, true_labels = evaluate(model, strat_model, dl_test, args.device)
    print(len(strat_li[0][0]), len(predictions[0][0]))
    with open(os.path.join(args.out_dir, f'{args.journalist}/predictions.pkl'), 'wb') as file:
        pickle.dump(predictions, file)

    with open(os.path.join(args.out_dir, f'{args.journalist}/strategies.pkl'), 'wb') as file:
        pickle.dump(strat_li, file)


    