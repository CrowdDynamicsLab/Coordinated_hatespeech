import json
import os
import pandas as pd
from datetime import datetime
import pickle 
from matplotlib import pyplot as plt
from collections import Counter
import numpy as np
import torch

class TreeNode:
    def __init__(self, name):
        self.name = name
        self.children = []  # List of TreeNode objects
        self.level = 0  # Level of the node in the tree
        self.sibling_order = 0  # Order among siblings
        self.parent = None  # Parent of the node
        self.local_relation = dict()

    def add_child(self, child_node):
        child_node.parent = self
        child_node.level = self.level + 1 if self.level is not None else 0
        child_node.sibling_order = len(self.children)
        self.children.append(child_node)

    def num_siblings(self):
        return len(self.parent.children)-1 if self.parent else 0
    
    def extract_data(node_list, only_leaf=False, f=lambda node: node.data):
        ret = []
        for node in node_list:
            if not (only_leaf and node.node_type == "type"):
                ret.append(f(node))
        return ret

    def create_local_relation(self):

        def _dfs(node):
            for child in node.children:
                node_child_rel = [child.level, child.num_siblings(), child.sibling_order]
                node_father_rel = [node.level, node.num_siblings(), node.sibling_order]
                #node_father_rel = child.parent
                node.local_relation[child.name] = [node_child_rel, node_father_rel, 0]
                child.local_relation[node.name] = [node_child_rel, node_father_rel, 1]
                _dfs(child)

        _dfs(self)

    def dfs(self):
        ret = []

        def _dfs(node, ret):
           #ret : List
            ret.append(node)
            for child in node.children:
                _dfs(child, ret)

        _dfs(self, ret)
        return ret
    
def build_tree(conversations):
    nodes = {}
    root = 0

    for parent, child in conversations:
        if parent not in nodes:
            nodes[parent] = TreeNode(parent)
        if child not in nodes:
            nodes[child] = TreeNode(child)

        nodes[parent].add_child(nodes[child])

        if not root:
            root = nodes[parent]

    return root

def get_node_info(tree_root):
    node_info = {}

    def traverse(node):
        node_info[node.name] = {
            'level': node.level,
            'number_of_siblings': node.num_siblings(),
            'sibling_order': node.sibling_order,
        }
        for child in node.children:
            traverse(child)

    traverse(tree_root)
    return node_info

# create conversation list as input into 'get_node_info'
def create_conversation_list(df, idx):
    conversations = []
    set_k = [idx]
    for _, row in df.iterrows():
        if idx != row['reference_id'] and [idx, row['reference_id']] not in conversations and row['reference_id'] not in df['tweet_id']:
            pair = [idx, row['reference_id']]
            conversations.append(pair)
            set_k.append(pair[0])
            set_k.append(pair[1])
        if row['reference_id'] not in set_k:
            pair = [idx, row['reference_id']]
            set_k.append(pair[1])
            conversations.append(pair)
        pair = [row['reference_id'], row['tweet_id']]
        set_k.append(pair[0])
        set_k.append(pair[1])
        conversations.append(pair)
    return conversations


def clamp_and_slice_ids(root_path, max_width, max_depth):
    """
        child_rel -> [0, 1, ..., max_width)
        ids \in [0, max_width - 1) do not change,
        ids >= max_width-1 are grouped into max_width-1
        apply this function in Node.extract_data(..., f=)
    """
    if max_width != -1:
        root_path = [min(ch_id, max_width - 1) for ch_id in root_path]
    if max_depth != -1:
        root_path = root_path[-max_depth:]
    return root_path


def generate_positions(root_paths, max_width, max_depth):
    """
    root_paths: List([ch_ids]), ch_ids \in [0, 1, ..., max_width)
    returns: Tensor [len(root_paths), max_width * max_depth]
    """
    for i, path in enumerate(root_paths):
        # stack-like traverse
        if len(root_paths[i]) > max_depth:
            root_paths[i] = root_paths[i][-max_depth:]
        root_paths[i] = [min(ch_id, max_width - 1) for ch_id in root_paths[i]]
        # pad
        root_paths[i] = root_paths[i][::-1] + [max_width] * (max_depth - len(root_paths[i]))
    root_path_tensor = torch.LongTensor(root_paths)
    onehots = torch.zeros((max_width + 1, max_width))
    onehots[:-1, :] = torch.eye(max_width)
    embeddings = torch.index_select(onehots, dim=0, index=root_path_tensor.view(-1))
    embeddings = embeddings.view(root_path_tensor.shape + (embeddings.shape[-1],))
    embeddings = embeddings.view((root_path_tensor.shape[0], -1))
    return embeddings

