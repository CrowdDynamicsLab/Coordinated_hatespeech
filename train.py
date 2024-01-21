# srun --mem=16g --nodes=1 --ntasks-per-node=1 --cpus-per-task=4 \
# --partition=gpuA100x4-interactive -interactive --account=bblr-delta-gpu \
# --gpus-per-node=1 --time=00:30:00 --x11 --pty /bin/bash
"""srun --account=bblr-delta-gpu --time=01:30:00 --nodes=1 --ntasks-per-node=16 \
--partition=gpuA100x4,gpuA40x4 --gpus=1 --mem=16g --pty /bin/bash"""
"""srun --account=bblr-delta-gpu --partition=gpuA100x4-interactive \
  --nodes=1 --gpus-per-node=1 --tasks=1 \
  --tasks-per-node=16 --cpus-per-task=1 --mem=20g \
  --pty bash"""
"""srun --account=bblr-delta-cpu --partition=cpu-interactive \
  --time=00:30:00 --mem=16g \
  jupyter-notebook --no-browser \
  --port=8991 --ip=0.0.0.0"""
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
from model import *


def load_data(data_dir, journalist, classes, batch_size, collate):
    ### Data (normalize input inter-event times, then padding to create dataloaders)
    num_classes, num_sequences = 0, 0
    seq_dataset = []
    arr = []
    dp = []
    rel = []
    
    split = [8, 12]
    val = 0
    journal_sort = pd.read_csv((os.path.join(data_dir, f'{journalist}/{journalist}_conv_labels.csv')))
    ids = list(set(journal_sort['conversation_id']))
    id_pair = {}
    id_conv = {}
    for idx in ids:
        id_pair[idx], id_conv[idx] = create_conversation_list(journal_sort[journal_sort['conversation_id']==idx], idx)
    id_data, data, label = create_data(journal_sort, ids)
    prob = pkl.load(open(os.path.join(data_dir, f'{journalist}/{journalist}_edgeprob.pkl'), 'rb'))
    
    with open(os.path.join(data_dir, f'{journalist}/{journalist}_global_path.txt'), "r") as f:
        for line in tqdm(f, total=get_number_of_lines(f)):
            dp.append(json.loads(line.strip()))

    with open(os.path.join(data_dir, f'{journalist}/{journalist}_local_path.txt'), "r") as f:
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
    model = CustomTransformerModel(args.feature_dim, args.global_dim, args.embed_size, args.num_classes, 
                               args.num_heads, args.num_layers, args.dropout, args.forward_expansion, args.max_len, args.mode)
    strat_model = StratModel(args.embed_size)
    model = model.to(args.device)
    strat_model = strat_model.to(args.device)
    
    logging.info(model)
    
    opt = torch.optim.Adam(model.parameters(), weight_decay=args.regularization, lr=args.learning_rate)
    model = model.to(args.device)
    strat_opt = torch.optim.Adam(strat_model.parameters(), lr=args.learning_rate)
    strat_model = strat_model.to(args.device)
    
    return model, opt, strat_model, strat_opt

def train(model, opt, strat_model, strat_opt, dl_train, dl_val, logging, max_epochs, patience, display_step, save_freq, out_dir, device, args, gmm = None):
    # Training (max_epochs or until the early stopping condition is satisfied)
    # Function that calculates the loss for the entire dataloader
    impatient = 0
    best_loss = np.inf
    best_model = deepcopy(model.state_dict())

    # for loop in range(args.max_loop):
    #     print('THIS IS LOOP', loop)
    #     best_in_this_loop = False
    #     impatient = 0
    #     if loop == 1:
    #         best_loss = np.inf
    #     if loop == 0:
    #         best_in_this_loop = True
    #     for epoch in range(max_epochs):
    #         # Train epoch
    #         model.train()
    #         for input in dl_train:
    #             #input = input.in_time.to(device), input.out_time.to(device),input.length.to(device), input.index.to(device),input.in_mark.to(device), input.out_mark.to(device)#move_input_batch_to_device(input, device)
    #             opt.zero_grad()
    #             loss = model(input.in_time.to(device), input.out_time.to(device),input.length.to(device),
    #                          input.index.to(device), input.in_mark.to(device), input.out_mark.to(device), 
    #                          input.in_tweet_type.to(device), input.out_tweet_type.to(device), 
    #                          use_marks, device)
    #             '''if use_marks:
    #                 log_prob, mark_nll, accuracy = model.log_prob(input)
    #                 loss = -model.module.aggregate(log_prob, input.length, device) + model.module.aggregate(
    #                     mark_nll, input.length, device)
    #                 del log_prob, mark_nll, accuracy
    #             else:
    #                 loss = -model.module.aggregate(model.module.log_prob(input), input.length, device)'''
    #             loss = (loss*sum_vec).sum()
    #             if loop != 0:
    #                 marks_set = set()
    #                 for batch in input.in_mark.tolist():
    #                     marks_set |= set(batch)
    #                 marks = torch.tensor(list(marks_set)).to(args.device)
    #                 marks_emb = model.rnn.mark_embedding(marks)
    #                 #print(marks_emb.size())
    #                 #print('loss_pre',loss)
    #                 loss += -gmm.score_samples(marks_emb).mean()/args.mark_embedding_size
    #                 #print('loss_post',loss)
    #             loss.backward()
    #             opt.step()
    #         # End of Train epoch

    #         model.eval()  # val losses over all val batches aggregated
    #         loss_val, loss_val_time, loss_val_marks, loss_val_acc = get_total_loss(
    #             dl_val, model, use_marks, device)
    #         loss_gmm = 0.0
    #         if loop != 0:
    #             loss_gmm = -gmm.score_samples(model.rnn.mark_embedding.weight.detach()).mean()/args.mark_embedding_size
    #             loss_val += loss_gmm
    #         plot_val_losses.append([loss_val, loss_val_time, loss_val_marks, loss_val_acc, loss_gmm])

    #         if (best_loss - loss_val) < 1e-4:
    #             impatient += 1
    #             if loss_val < best_loss:
    #                 best_loss = loss_val
    #                 best_model = deepcopy(model.state_dict())
    #                 if not best_in_this_loop:
    #                     best_in_this_loop = True
    #                     best_gmm = deepcopy(gmm.state_dict())
    #         else:
    #             best_loss = loss_val
    #             best_model = deepcopy(model.state_dict())
    #             impatient = 0
    #             if not best_in_this_loop:
    #                 best_in_this_loop = True
    #                 best_gmm = deepcopy(gmm.state_dict())

    #         if impatient >= patience:
    #             logging.info(f'Breaking due to early stopping at epoch {epoch}'); break

    #         if (epoch + 1) % display_step == 0:
    #             amdn_loss = loss_val-loss_gmm
    #             logging.info(f"Epoch {epoch+1:4d}, trlast = {loss:.4f}, val = {loss_val:.4f}, amdn_loss = {amdn_loss:.4f}, gmmval = {loss_gmm:.4f}")
            
    #         if (epoch + 1) % save_freq == 0:
    #             if loop == 0:
    #                 torch.save(best_model, os.path.join(out_dir, 'best_pre_train_model_state_dict_ep_{}.pt'.format(epoch)))
    #                 # evaluate(model, [dl_train, dl_val], ['Ckpt_train', 'Ckpt_val'], use_marks, device)
    #                 logging.info(f"saved intermediate pre-trained checkpoint")
    #             else:
    #                 torch.save(best_model, os.path.join(out_dir, 'best_model_state_dict_iter_{}_ep_{}.pt'.format(loop, epoch)))
    #                 # evaluate(model, [dl_train, dl_val], ['Ckpt_train', 'Ckpt_val'], use_marks, device)
    #                 logging.info(f"saved intermediate checkpoint")
        
    criterion = nn.CrossEntropyLoss()
    min_loss = float('inf')
    best_model_path = os.path.join(f'{args.journalist}/best_model_w_strat.pth') 
    best_strat_model_path = os.path.join(f'{args.journalist}/best_strat_model_w_strat.pth') 
    use_strat = args.use_strat
    for epoch in range(args.max_epochs):  # Number of epochs
        pred_tr = []
        true_tr = []
        prob_tr = []
        for item in dl_train:
            # Forward pass
            data = item.data.float().to(args.device)
            dp = item.global_path.float().to(args.device)
            rel = item.local_path.float().to(args.device)
            prob = item.prob.float().to(args.device)
            targets = item.labels.long().to(args.device)
            mask = item.masks.float().to(args.device)
            
            output, p_output = model(data, dp, rel, mask)
            prob_tr.append(p_output)
            prob_output = strat_model(p_output.detach())
            _, predicted = torch.max(output.data, 2)
            pred_tr.extend(predicted.view(-1).tolist())
            true_tr.extend(targets.view(-1).tolist())
            correct_tr = sum(p == t for p, t in zip(pred_tr, true_tr))
            acc_tr = correct_tr / len(true_tr)
            loss = criterion(output.view(-1, args.num_classes), targets.view(-1))
            
            # Apply the padding mask
            loss = loss * mask
            
            # Compute the mean loss, considering only non-padded elements
            loss = loss.sum() / mask.sum()

            # Backward and optimize
            opt.zero_grad()
            loss.backward()
            opt.step()

            if use_strat == True:
                # Dot product between probability_output and prob
                dot_product = torch.sum(prob_output * prob, dim=-1)  # Sum over the last dimension
                # Taking negative logarithm; adding a small value for numerical stability
                log_loss = -torch.log(dot_product + 1e-9)  
                # Masking and averaging the additional loss
                loss_strat = (log_loss * mask).sum() / mask.sum()
                # Combine the losses
                total_loss = loss + loss_strat
                strat_opt.zero_grad()
                loss_strat.backward()  # No need to retain graph here
                strat_opt.step()
                #total_loss.backward()
                #optimizer.step()

            else:
                total_loss = loss
                #total_loss.backward()
                #optimizer.step()
            
        if loss.item() < min_loss:
            min_loss = loss.item()
            print(f"Epoch [{epoch+1}/10], New Min Loss: {min_loss}, New Strategy Loss: {total_loss}, Acc: {acc_tr}")
            # Save the model state
            torch.save(model.state_dict(), os.path.join(args.out_dir, best_model_path))
            torch.save(strat_model.state_dict(), os.path.join(args.out_dir, best_strat_model_path))
            evaluate(model, dl_val, args.device)
            

    logging.info('Training finished.............')
    model.load_state_dict(torch.load(os.path.join(args.out_dir, best_model_path)))
    strat_model.load_state_dict(torch.load(os.path.join(args.out_dir, best_strat_model_path)))
    torch.save(model, os.path.join(args.out_dir, best_model_path))
    logging.info(f"The entire model is saved in {os.path.join(args.out_dir, best_model_path)}.")    
    # loading model model = torch.load(save_model_path)
    
    # Plot training curve displaying separated validation losses
    # plot_loss = np.array(torch.tensor(plot_val_losses).detach().cpu())
    # if len(plot_loss) > patience:
    #     plot_loss = plot_loss[:-patience] # plot only until early stopping
    # fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    # plot_labels = ['Total_loss', 'Time_NLL', 'Marks_NLL', 'Marks_Acc', 'GMM_Loss']
    # for i in range(plot_loss.shape[1]):
    #     ax = axes[i]
    #     ax.plot(range(len(plot_loss)), plot_loss[:, i], marker='o', label=plot_labels[i], markersize=3)
    #     ax.set_xlabel('Val Loss vs. Training Epoch')
    #     # ax.set_ylabel(plot_labels[i])
    #     # ax.set_title('Validation dataset')
    #     ax.legend()
    # plt.savefig(os.path.join(out_dir, 'training_curve.png'))
    

def evaluate(model, dl_test, device):
        # Calculate the train/val/test loss, plot training curve
        model.eval()
        strat_model.eval()
        with torch.no_grad():
            test_loss = 0
            test_steps = 0
            predictions = []
            true_labels = []
            strat_li = []
            for item in tqdm(dl_test):  # Assuming 'val' is your validation dataset
                # Forward pass
                data = item.data.float().to(device)
                dp = item.global_path.float().to(device)
                rel = item.local_path.float().to(device)
                targets = item.labels.long().to(device)
                mask = item.masks.float().to(device)

                output, p_output = model(data, dp, rel, mask)
                prob_output = strat_model(p_output.detach())
                strat_li.append(prob_output)
                _, predicted = torch.max(output.data, 2)
                predictions.extend(predicted.view(-1).tolist())
                true_labels.extend(targets.view(-1).tolist())

            correct_predictions = sum(p == t for p, t in zip(predictions, true_labels))
            accuracy = correct_predictions / len(true_labels)
            print(f'Test Accuracy: {accuracy:.4f}')
            #logging.info(f'{dl_name}: {loss_tot:.4f}')
            logging.info(f'Test Accuracy: {accuracy:.4f}')
        return predictions, strat_li, true_labels




if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Tempt model')
    
    
    ## dataset and output directories
    parser.add_argument('--data_dir', type=str, default='../data')
    parser.add_argument('--out_dir', type=str, default='../data')
    parser.add_argument('--log_filename', type=str, default='run.log')
    parser.add_argument('--journalist', type=str, default='sallykohn')
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
    logging.info('Logging any runs of this program - appended to same file.')
    logging.info('Arguments = {}'.format(args))
    dl_train, dl_val, dl_test = load_data(args.data_dir, args.journalist, args.classes, args.batch_size, collate)
    
    logging.info('loaded the dataset and formed torch dataloaders.')

    
    model, opt, strat_model, strat_opt = create_model(args.classes, args)
    logging.info('model created from config hyperparameters.')

    train(model, opt, strat_model, strat_opt, dl_train, dl_val, logging, args.max_epochs, args.patience, 
          args.display_step, args.save_freq, args.out_dir, args.device, args)

    dl_list = dl_val #[dl_train, dl_val, dl_test]
    dl_names = ['Train', 'Val', 'Test']
    evaluate(model, dl_test, args.device)

    best_model_path = os.path.join(f'{args.journalist}/best_model_w_strat.pth') 
    best_strat_model_path = os.path.join(f'{args.journalist}/best_strat_model_w_strat.pth') 

    model = torch.load(os.path.join(args.out_dir, best_model_path))
    strat_model = torch.load(os.path.join(args.out_dir, best_strat_model_path))
    
    # extract_features(model, logging, args, gmm)
    # logging.info('Finished program.')
    