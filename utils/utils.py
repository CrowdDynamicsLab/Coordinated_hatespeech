#!/usr/bin/env python3
# modified utils.py from Code Prediction By Feeding Trees To Transformers 
# https://arxiv.org/abs/2003.13848

import math
import multiprocessing as mp

from tqdm import tqdm

import os
import sys
import random
import numpy as np
import re
import ast

from collections import OrderedDict
from time import gmtime, strftime
from scipy.sparse import csr_matrix 

#from . import constants

def pad_sequences(sequences, pad_token=0):
    # Determine the maximum sequence length
    max_length = max(len(seq) for seq in sequences)

    # Pad each sequence to the maximum length
    padded_sequences = np.array([np.pad(seq, ((0, max_length - len(seq)), (0, 0)), 
                                        mode='constant', constant_values=pad_token) 
                                 for seq in sequences])

    # Create attention masks
    attention_masks = np.array([[1 if token.any() else 0 for token in seq] 
                                for seq in padded_sequences])
    
    return padded_sequences, attention_masks

def convert_path(g_list):
    g_dict = {}
    for item in g_list:
        temp = ast.literal_eval(item)
        for item in temp:
            k = list(item.keys())[0]
            g_dict[k] = item[k][k]
    return g_dict

def convert_global(root_paths, id_data):
    roots = []
    global_new = []
    for i in range(len(root_paths)):
        new_dict = {}
        for item in root_paths[i]:
            name = list(item.keys())[0]
            new_dict[name] = np.array(list(list(item.values())[0].values())).squeeze().tolist()
        roots.append(new_dict)
    for i in range(len(id_data)):
        global_new.append([roots[i][str(k)] for k in id_data[i]])
    return global_new

def convert_local(local_rel):
    rel = []
    for i in range(len(local_rel)):
        new_dict = {}
        for item in local_rel[i]:
            name = list(item.keys())[0]
            new_dict[name] = item[name]
        rel.append(new_dict)
    return rel

def create_data(journal_sort, ids):
    batch_data = []
    target_data = []
    conv_data = []
    ref_data = []
    id_data = []
    for idx in ids:
        convs = journal_sort[journal_sort['conversation_id'] == idx]
        convs_batch = convs[['type', 'possibly_sensitive', 'lang', 'reply_settings', 
                               'retweet_count', 'reply_count', 'like_count', 'quote_count',
                                'impression_count', 'mentions', 'urls']]
        #conv_data.append(list(convs['conversation_id']))
        conv_data.append(convs_batch.to_numpy().tolist())
        ref_data.append(list(convs['reference_id']))
        id_data.append(list(convs['tweet_id']))
        batch_data.append(convs_batch.values.tolist())
        target_data.append(list(convs['labels']))
    
    label_data = target_data
    return id_data, conv_data, label_data

def create_mat(local_mat, mat_type):
    result = []
    for ind, item in enumerate(local_mat):
        max_row = max(i[0] for i in item)+1
        max_col = max(i[1] for i in item)+1
        if mat_type == 'sum':
            row = np.array(item)[:,0]
            col = np.array(item)[:,1]

            # taking data 
            data = np.array([sum(np.array(i)[2]) for i in item])

            # creating sparse matrix 
            sparseMatrix = csr_matrix((data, (row, col)), shape = (dim, dim)).toarray() 
            result.append(sparseMatrix)
        else:
            matrix = np.zeros((max_row, max_col, 3), dtype=float)
            for x in item:
                row, col, value = x
                matrix[row, col] = [i + 0.05 for i in value]
            result.append(matrix)
    return np.array(result)

def indexing(ls):
    dic = {}
    for i in range(len(ls)):
        dic[ls[i]] = i
        i += 1
    return dic

def generate_local_mat(local, idx):
    mat = []
    for ids, item in enumerate(local):
        #print(ids)
        temp = []
        ind = indexing(idx[ids])
        for i in idx[ids]:
            if str(i) not in list(local[ids].keys()):
                continue
            for k in list(local[ids][str(i)].keys()):
                if k == list(local[ids].keys())[0]:
                    temp_l = local[ids][str(i)][k]
                    temp_ind = ind[i]
                    temp.append([temp_ind, temp_ind, temp_l[temp_l[2]]])
                elif int(k) not in idx[ids]:
                    continue
                else:
                    temp_l = local[ids][str(i)][k]
                    temp_ind1 = ind[i]
                    temp_ind2 = ind[int(k)]
                    temp.append([temp_ind1, temp_ind2, temp_l[temp_l[2]]])
        if not temp:
            for i in idx[ids]:
                temp_ind = ind[i]
                temp.append([temp_ind, temp_ind, [0, 0, 0]])
        mat.append(temp)
    return mat

def line_positions(file_path):
    with open(file_path) as f:
        while True:
            pos = f.tell()
            if f.readline():
                yield pos
            else:
                break

def get_number_of_lines(fobj):
    nol = sum(1 for _ in fobj)
    fobj.seek(0)
    return nol

def file_tqdm(f, use_tqdm=False):
    if use_tqdm:
        return tqdm(f, total=get_number_of_lines(f))
    else:
        return f

def parallelize(iterable, f, f_args=(), worker_init=None, n_cores=None):
    if n_cores == 1:
        return _mp_iterate_over(f, iterable, f_args)
    if n_cores is None:
        n_cores = int(mp.cpu_count())
    lst = list(iterable)
    chunksize = math.ceil(len(lst) / n_cores)
    with mp.Pool(processes=n_cores, initializer=worker_init) as pool:
        jobs = [
            pool.apply_async(
                _mp_iterate_over, (f, lst[i * chunksize: (i + 1) * chunksize], f_args)
            )
            for i in range(n_cores)
        ]
        multiple_results = [job.get() for job in jobs]
        results = flatten(multiple_results)
    return results

def _mp_iterate_over(f, lst, f_args):
    return [f(x, *f_args) for x in lst]

def flatten(list_of_lists):
    return [x for xs in list_of_lists for x in xs]


########################################################################
# generating dataset utils        

def get_dfs(ast, only_leaf=False):
    dp = []
    for node in ast:
        if "value" in node:
            dp.append(node["value"])
        else:
            if not only_leaf:
                dp.append(node["type"])
    return dp

def separate_dps(ast, max_len):
    """
    Handles training / evaluation on long ASTs by splitting
    them into smaller ASTs of length max_len, with a sliding
    window of max_len / 2.

    Example: for an AST ast with length 1700, and max_len = 1000,
    the output will be:
    [[ast[0:1000], 0], [ast[500:1500], 1000], [ast[700:1700], 1500]]

    Input:
        ast : List[Dictionary]
            List of nodes in pre-order traversal.
        max_len : int

    Output:
        aug_asts : List[List[List, int]]
            List of (ast, beginning idx of unseen nodes)
    """
    half_len = int(max_len / 2)
    if len(ast) <= max_len:
        return [[ast, 0]]

    aug_asts = [[ast[:max_len], 0]]
    i = half_len
    while i < len(ast) - max_len:
        aug_asts.append([ast[i: i + max_len], half_len])
        i += half_len
    idx = max_len - (len(ast) - (i + half_len))
    aug_asts.append([ast[-max_len:], idx])

    return aug_asts

def separate_lrs(lrs, max_len):
    def reformat(lrs, left):  # [left,right)
        new_lrs = []
        for idx, lr in enumerate(lrs):
            # lr -> dict: {idx:[],idx:[]}
            temp_lr = dict()
            for key, val in lr.items():
                if left <= key < left + max_len:
                    temp_lr[key - left] = val
            new_lrs.append(temp_lr)
        return new_lrs

    half_len = int(max_len / 2)
    if len(lrs) <= max_len:
        return [[reformat(lrs, 0), 0]]

    aug_asts = [[reformat(lrs[:max_len], 0), 0]]
    i = half_len
    while i < len(lrs) - max_len:
        aug_asts.append([reformat(lrs[i: i + max_len], i), half_len])
        i += half_len
    idx = max_len - (len(lrs) - (i + half_len))
    aug_asts.append([reformat(lrs[len(lrs) - max_len:], len(lrs) - max_len), idx])
    return aug_asts

def separate_types_values(dp, mode):
    """
        constructs two separate sequence of types and values
        if node do not contain value, sets constants.EMPTY token
    """

    def copy_if_key(tgt, src, key, default=None):
        if key in src:
            tgt[key] = src[key]
        elif default is not None:
            tgt[key] = default

    types = []
    values = []
    for i, node in enumerate(dp):
        val = {}
        copy_if_key(val, node, "children")
        copy_if_key(val, node, "value", constants.EMPTY)
        values.append(val)
        if mode == "all":
            typ = {}
            copy_if_key(typ, node, "children")
            copy_if_key(typ, node, "type")
            types.append(typ)
    return (types, values)


def get_ancestors(ast):
    ancestors = {0: []}
    node2parent = {0: 0}
    for i, node in enumerate(ast):
        if "children" in node:
            for child in node["children"]:
                node2parent[child] = i
        ancestors[i] = [i] + ancestors[node2parent[i]]
    return ancestors


def get_terminal_nodes(ast):
    terminal_nodes = [i for i, node in enumerate(ast) if "children" not in node]
    return terminal_nodes


def tokenize(s):
    pattern = re.compile(r"(?<!^)(?=[A-Z])")
    tokenized = pattern.sub("_", s).lower().split("_")
    return list(filter(None, tokenized))[:5]


letters = "abcdefghijklmnopqrstuvwxyz"

##########

########
# https://stackoverflow.com/questions/47776486/python-struct-error-i-format-requires-2147483648-number-2147483647
# only needed for parallizing tree relative attention matrices calculation
import functools
import logging
import struct
import sys

logger = logging.getLogger()


def patch_mp_connection_bpo_17560():
    """Apply PR-10305 / bpo-17560 connection send/receive max size update
    See the original issue at https://bugs.python.org/issue17560 and 
    https://github.com/python/cpython/pull/10305 for the pull request.
    This only supports Python versions 3.3 - 3.7, this function
    does nothing for Python versions outside of that range.
    """
    patchname = "Multiprocessing connection patch for bpo-17560"
    if not (3, 3) < sys.version_info < (3, 8):
        logger.info(
            patchname + " not applied, not an applicable Python version: %s",
            sys.version
        )
        return

    from multiprocessing.connection import Connection

    orig_send_bytes = Connection._send_bytes
    orig_recv_bytes = Connection._recv_bytes
    if (
            orig_send_bytes.__code__.co_filename == __file__
            and orig_recv_bytes.__code__.co_filename == __file__
    ):
        logger.info(patchname + " already applied, skipping")
        return

    @functools.wraps(orig_send_bytes)
    def send_bytes(self, buf):
        n = len(buf)
        if n > 0x7fffffff:
            pre_header = struct.pack("!i", -1)
            header = struct.pack("!Q", n)
            self._send(pre_header)
            self._send(header)
            self._send(buf)
        else:
            orig_send_bytes(self, buf)

    @functools.wraps(orig_recv_bytes)
    def recv_bytes(self, maxsize=None):
        buf = self._recv(4)
        size, = struct.unpack("!i", buf.getvalue())
        if size == -1:
            buf = self._recv(8)
            size, = struct.unpack("!Q", buf.getvalue())
        if maxsize is not None and size > maxsize:
            return None
        return self._recv(size)

    Connection._send_bytes = send_bytes
    Connection._recv_bytes = recv_bytes

    print(patchname + " applied")

##########
