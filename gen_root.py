# reads asts
# generates tree positional embeddings data for the datapoints
# from the paper "Novel positional encodings to enable tree-based transformers"


import argparse
import json
import logging
import os

import sys

sys.setrecursionlimit(10000)

import pandas as pd

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
from utils.utils import separate_dps, file_tqdm, separate_lrs
from utils.tree_utils import *
from dataset import *

logging.basicConfig(level=logging.INFO)


def main():
    parser = argparse.ArgumentParser(description="Generate datapoints from AST")
    parser.add_argument("--data_dir", "-o", default="../data", help="filepath with the ASTs to be parsed")
    parser.add_argument("--journalist", type=str, default="muyixiao", help="filepath with the ASTs to be parsed")
    parser.add_argument(
        "--out_dir", default="../data/", help="filepath for the output dps"
    )
    parser.add_argument(
        "--n_ctx", "-c", type=int, default=2000, help="max context length for each dp"
    )
    parser.add_argument(
        "--max_width", type=int, default=16, help="max number of child ids"
    )
    parser.add_argument(
        "--max_depth", type=int, default=10, help="max depth of the leaf to root path"
    )

    # HERE
    parser.add_argument("--td", action="store_true", help="Advance Setting, store both child path and father path")
    parser.add_argument("--local_relation", action="store_true", help="")

    args = parser.parse_args()
    
    logging.info("Number of context: {}".format(args.n_ctx))

    num_dps = 0
    #data = pkl.load(open(os.path.join(args.data_dir, f'{args.journalist}_dict.pkl'), 'rb'))
    num_classes = 3
    #num_sequences = len(set(data['conversation_id']))
    #journal = pd.DataFrame.from_dict(data)
    #journal_sort = journal.sort_values(by=['created_at'])
    journal_sort = pd.read_csv(os.path.join(args.data_dir, f'{args.journalist}/{args.journalist}_conv_labels.csv'))

    journal_batch = journal_sort[["type", "possibly_sensitive", "lang", "reply_settings",
                                "retweet_count", "reply_count", "like_count", "quote_count", "impression_count",
                                "mentions", "urls", "labels"]]
    ids = list(set(journal_sort['conversation_id']))
    id_pair = {}
    for idx in ids:
        id_pair[idx] = create_conversation_list(journal_sort[journal_sort['conversation_id']==idx], idx)

    if args.local_relation:
        print("Save Relation between Child and Father")
        with open(os.path.join(args.out_dir, f'{args.journalist}/{args.journalist}_local_path.txt'), "w") as fout:
            for k in id_pair.keys():
                tree_root = build_tree(id_pair[k])
                
                tree_root.create_local_relation()
                node_list = tree_root.dfs()
                local_relations = TreeNode.extract_data(node_list,f=lambda node: node.local_relation)
                lrs = separate_lrs(local_relations, args.n_ctx)

                for lr, extended in lrs:
                    if extended != 0:
                        break
                    if len(lr) - extended > 1:
                        json.dump(lr, fp=fout)  # each line is the json of a list [dict,dict,...]
                        num_dps += 1
                        fout.write("\n")

    else:
        if args.td:
            print("Save Child Paths and Father Paths")

        num_dps = 0
        with open(os.path.join(args.out_dir, f'{args.journalist}_global_path.txt'), "w") as fout:
            for k in id_pair.keys():
                tree_root = build_tree(id_pair[k])
                
                tree_root.create_global_relation()
                node_list = tree_root.dfs()
                
                root_paths = TreeNode.extract_data(node_list,f=lambda node: clamp_and_slice_ids(
                        node.global_relation, max_width=-1, max_depth=-1))
                asts = separate_dps(root_paths, args.n_ctx)

                for lr, extended in asts:
                    if extended != 0:
                        break
                    if len(lr) - extended > 1:
                        json.dump(lr, fp=fout)  # each line is the json of a list [dict,dict,...]
                        num_dps += 1
                        fout.write("\n")


        logging.info("Wrote {} data points to: {}".format(num_dps, args.out_dir))


if __name__ == "__main__":
    main()
