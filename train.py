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


def load_data(data_dir, journalist, classes, batch_size, collate):
    ### Data (normalize input inter-event times, then padding to create dataloaders)
    num_classes, num_sequences = 0, 0
    seq_dataset = []
    arr = []
    dp = []
    rel = []
    
    split = [64, 128]
    val = 0
    journal_sort = pd.read_csv((os.path.join(data_dir, f'{journalist}_context.csv')))
    ids = list(set(journal_sort['conversation_id']))
    id_pair = {}
    id_conv = {}
    for idx in ids:
        id_pair[idx], id_conv[idx] = create_conversation_list(journal_sort[journal_sort['conversation_id']==idx], idx)
    id_data, data, label = create_data(journal_sort, ids)
    prob = pkl.load(open(os.path.join(data_dir, f'{journalist}_edgeprob.pkl'), 'rb'))
    
    with open(os.path.join(data_dir, f'{journalist}_global_path.txt'), "r") as f:
        for line in tqdm(f, total=get_number_of_lines(f)):
            dp.append(json.loads(line.strip()))

    with open(os.path.join(data_dir, f'{journalist}_local_path.txt'), "r") as f:
        for line in tqdm(f, total=get_number_of_lines(f)):
            rel.append(json.loads(line.strip()))
    
    global_input = convert_global(dp, id_data)
    local_data = convert_local(rel)
    local_mat = generate_local_mat(local_data, id_data)
    local_input = create_mat(local_mat, mat_type='concat')
    logging.info(f'loaded split {journalist}...')
    # data - dict: dim_process, devtest, args, train, dev, test, index (train/dev/test given as)
    # data[split] - list dicts {'time_since_start': at, 'time_since_last_event': dt, 'type_event': mark} or
    # data[split] - dict {'arrival_times', 'delta_times', 'marks'}
    # data['dim_process'] = Number of accounts = 119,298
    # num_sequences: number of conversations of a journalist
    num_classes = classes
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
    dl_train = torch.utils.data.DataLoader(d_train, batch_size=batch_size, shuffle=False, collate_fn=collate)
    dl_val = torch.utils.data.DataLoader(d_val, batch_size=batch_size, shuffle=False, collate_fn=collate)
    dl_test = torch.utils.data.DataLoader(d_test, batch_size=batch_size, shuffle=False, collate_fn=collate)
    return dl_train, dl_val, dl_test

def create_model(num_classes, args):
    # General model config
    # general_config = dpp.model.ModelConfig(
    #     encoder_type=args.encoder_type, use_history=args.use_history, history_size=args.history_size, rnn_type=args.rnn_type,
    #     use_embedding=args.use_embedding, embedding_size=args.embedding_size, num_embeddings=num_sequences, # seq emb
    #     use_marks=args.use_marks, mark_embedding_size=args.mark_embedding_size, num_classes=num_classes,
    #     heads=args.heads, depth=args.depth, wide=args.wide, seq_length=args.max_seq_length, device=args.device,
    #     pos_enc=args.pos_enc, add=args.add, time_opt=args.time_opt, expand_dim=args.expand_dim,
    # )
                    
    # Decoder specific config
    model = model.TransformerModel(args.d_model, args.num_heads, args.num_layers, arg.d_ff, args.max_seq_length, args.dropout)
    # model = nn.DataParallel(dpp.model.Model(general_config, decoder)).to(args.device)
    model = model.to(args.device)
    logging.info(model)
    
    opt = torch.optim.Adam(model.parameters(), weight_decay=args.regularization, lr=args.learning_rate)
    model = model.to(args.device)
    #opt = torch.optim.Adam(model.module.parameters(), weight_decay=args.regularization, lr=args.learning_rate)
    #opt = torch.nn.DataParallel(opt,[1,2,3])
    # for name, param in model.named_parameters():
    #    logging.info(name, param.device)
    
    return model, opt

def train(model, opt, dl_train, dl_val, logging, use_marks, 
          max_epochs, patience, display_step, save_freq, out_dir, device, args, gmm = None):
    # Training (max_epochs or until the early stopping condition is satisfied)
    # Function that calculates the loss for the entire dataloader
    #gmm = GaussianTiedMixture(args.gmm_k, args.mark_embedding_size, device = device)
    #gmm.to(device)
    impatient = 0
    best_loss = np.inf
    best_model = deepcopy(model.state_dict())
    best_gmm = deepcopy(gmm.state_dict())
    plot_val_losses = []
    sum_vec = [args.gpu0sz/args.batch_size]
    for i in range(3):
        sum_vec.append((args.batch_size-args.gpu0sz)/(3*args.batch_size))
    sum_vec = torch.tensor(sum_vec).to(device)
    for loop in range(args.max_loop):
        print('THIS IS LOOP', loop)
        best_in_this_loop = False
        impatient = 0
        if loop == 1:
            best_loss = np.inf
        if loop == 0:
            best_in_this_loop = True
        for epoch in range(max_epochs):
            # Train epoch
            model.train()
            for input in dl_train:
                #input = input.in_time.to(device), input.out_time.to(device),input.length.to(device), input.index.to(device),input.in_mark.to(device), input.out_mark.to(device)#move_input_batch_to_device(input, device)
                opt.zero_grad()
                loss = model(input.in_time.to(device), input.out_time.to(device),input.length.to(device),
                             input.index.to(device), input.in_mark.to(device), input.out_mark.to(device), 
                             input.in_tweet_type.to(device), input.out_tweet_type.to(device), 
                             use_marks, device)
                '''if use_marks:
                    log_prob, mark_nll, accuracy = model.log_prob(input)
                    loss = -model.module.aggregate(log_prob, input.length, device) + model.module.aggregate(
                        mark_nll, input.length, device)
                    del log_prob, mark_nll, accuracy
                else:
                    loss = -model.module.aggregate(model.module.log_prob(input), input.length, device)'''
                loss = (loss*sum_vec).sum()
                if loop != 0:
                    marks_set = set()
                    for batch in input.in_mark.tolist():
                        marks_set |= set(batch)
                    marks = torch.tensor(list(marks_set)).to(args.device)
                    marks_emb = model.rnn.mark_embedding(marks)
                    #print(marks_emb.size())
                    #print('loss_pre',loss)
                    loss += -gmm.score_samples(marks_emb).mean()/args.mark_embedding_size
                    #print('loss_post',loss)
                loss.backward()
                opt.step()
            # End of Train epoch

            model.eval()  # val losses over all val batches aggregated
            loss_val, loss_val_time, loss_val_marks, loss_val_acc = get_total_loss(
                dl_val, model, use_marks, device)
            loss_gmm = 0.0
            if loop != 0:
                loss_gmm = -gmm.score_samples(model.rnn.mark_embedding.weight.detach()).mean()/args.mark_embedding_size
                loss_val += loss_gmm
            plot_val_losses.append([loss_val, loss_val_time, loss_val_marks, loss_val_acc, loss_gmm])

            if (best_loss - loss_val) < 1e-4:
                impatient += 1
                if loss_val < best_loss:
                    best_loss = loss_val
                    best_model = deepcopy(model.state_dict())
                    if not best_in_this_loop:
                        best_in_this_loop = True
                        best_gmm = deepcopy(gmm.state_dict())
            else:
                best_loss = loss_val
                best_model = deepcopy(model.state_dict())
                impatient = 0
                if not best_in_this_loop:
                    best_in_this_loop = True
                    best_gmm = deepcopy(gmm.state_dict())

            if impatient >= patience:
                logging.info(f'Breaking due to early stopping at epoch {epoch}'); break

            if (epoch + 1) % display_step == 0:
                amdn_loss = loss_val-loss_gmm
                logging.info(f"Epoch {epoch+1:4d}, trlast = {loss:.4f}, val = {loss_val:.4f}, amdn_loss = {amdn_loss:.4f}, gmmval = {loss_gmm:.4f}")
            
            if (epoch + 1) % save_freq == 0:
                if loop == 0:
                    torch.save(best_model, os.path.join(out_dir, 'best_pre_train_model_state_dict_ep_{}.pt'.format(epoch)))
                    # evaluate(model, [dl_train, dl_val], ['Ckpt_train', 'Ckpt_val'], use_marks, device)
                    logging.info(f"saved intermediate pre-trained checkpoint")
                else:
                    torch.save(best_model, os.path.join(out_dir, 'best_model_state_dict_iter_{}_ep_{}.pt'.format(loop, epoch)))
                    # evaluate(model, [dl_train, dl_val], ['Ckpt_train', 'Ckpt_val'], use_marks, device)
                    logging.info(f"saved intermediate checkpoint")
        model.load_state_dict(best_model)
        ## fitting gmm
        #gmm.to(args.device)
        if loop < args.max_loop:
            print('fitting gmm')
            if loop == 0:
                z = model.rnn.mark_embedding.weight.cpu().detach().numpy()
                km = KMeans(n_clusters=args.gmm_k, random_state=0, n_init=50, max_iter=500).fit(z)
                
                #km.fit(z)
                
                mu_init=torch.tensor(km.cluster_centers_).unsqueeze(0)
                #var_init=torch.tensor(tmpGmm.covariances_).unsqueeze(0).unsqueeze(0)
                gmm.mu_init = mu_init
                #gmm.var_init = var_init
                gmm._init_params()
                #pi = np.concatenate((tmpGmm.weights_[g_cls],tmpGmm.weights_[l_cls]), axis = 0)
                #gmm.pi.data = torch.tensor(pi).unsqueeze(0).unsqueeze(-1).to(args.device)
            gmm.fit(model.rnn.mark_embedding.weight.detach().to(args.device), warm_start=True)
            print(-gmm.score_samples(model.rnn.mark_embedding.weight.detach().to(args.device)).mean()/args.mark_embedding_size)
        
    logging.info('Training finished.............')
    torch.save(best_model, os.path.join(out_dir, 'best_model_state_dict.pt'))
    torch.save(best_gmm, os.path.join(out_dir, 'best_gmm.pt'))
    model.load_state_dict(best_model)
    torch.save(model, os.path.join(out_dir, 'best_full_model.pt'))
    logging.info(f"The entire model is saved in {os.path.join(out_dir, 'best_full_model.pt')}.")    
    # loading model model = torch.load(save_model_path)
    
    # Plot training curve displaying separated validation losses
    plot_loss = np.array(torch.tensor(plot_val_losses).detach().cpu())
    if len(plot_loss) > patience:
        plot_loss = plot_loss[:-patience] # plot only until early stopping
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    plot_labels = ['Total_loss', 'Time_NLL', 'Marks_NLL', 'Marks_Acc', 'GMM_Loss']
    for i in range(plot_loss.shape[1]):
        ax = axes[i]
        ax.plot(range(len(plot_loss)), plot_loss[:, i], marker='o', label=plot_labels[i], markersize=3)
        ax.set_xlabel('Val Loss vs. Training Epoch')
        # ax.set_ylabel(plot_labels[i])
        # ax.set_title('Validation dataset')
        ax.legend()
    plt.savefig(os.path.join(out_dir, 'training_curve.png'))
    







if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Tempt model')
    
    
    ## dataset and output directories
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--out_dir', type=str, default='./data/result')
    parser.add_argument('--log_filename', type=str, default='run.log')
    parser.add_argument('--journalist', type=str, default='aliceysu')
    parser.add_argument('--classes', type=int, default=3)

    ## model encoder parameters
    parser.add_argument('--depth', type=int, default=1)
    parser.add_argument('--wide', dest='wide', default=True, action='store_true', help='Change back')
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--num_layers', type=int, default=3, help='flow-based models.')
    parser.add_argument('--d_ff', type=int, default=None)
    parser.add_argument('--max_seq_length', type=int, default=2000)
    parser.add_argument('--dropout', type=float, default=0.1)

    
    
    ## training arguments
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--regularization', type=float, default=1e-5)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    
    
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
    dl_train, dl_val, dl_test = load_data(args.data_dir, args.journalist, args.classes, args.batch_size, collate)
    
    logging.info('loaded the dataset and formed torch dataloaders.')

    
    model, opt = create_model(args.classes, args)
    logging.info('model created from config hyperparameters.')

    train(model, opt, dl_train, dl_val, logging, args.use_marks, args.max_epochs, args.patience, 
          args.display_step, args.save_freq, args.out_dir, args.device, args)

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
    