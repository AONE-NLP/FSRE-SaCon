
import json 
import random
import os 
import sys 
sys.path.append("..")
import pdb 
import re 
import pdb 
import math 
import torch
import numpy as np  
from collections import Counter
from torch.utils import data
sys.path.append("../../")
from utils.utils import EntityMarker


class SaConDataset(data.Dataset):
    """Overwritten class Dataset for model SaCon.

    This class prepare data for training of SaCon.
    """
    def __init__(self, path, args):
        """Inits tokenized sentence and positive pair for SaCon.
        
        Args:
            path: path to your dataset.
            args: args from command line.
        
        Returns:
            No returns
        
        Raises:
            If the dataset in `path` is not the same format as described in 
            file 'prepare_data.py', there may raise:
                - `key nor found`
                - `integer can't be indexed`
                and so on.
        """
        self.path = path 
        self.args = args 
        data = json.load(open(os.path.join(path, "cpdata.json")))
        entityMarker = EntityMarker()

        self.tokens = np.zeros((len(data), args.max_length), dtype=int)
        self.mask = np.zeros((len(data), args.max_length), dtype=int)
        self.h_pos = np.zeros((len(data)), dtype=int)
        self.t_pos = np.zeros((len(data)), dtype=int)
        self.rel_tokens = np.zeros((len(data), args.max_length), dtype=int)
        self.rel_mask = np.zeros((len(data), args.max_length), dtype=int)

        self.len = len(data)


        for i, sentence in enumerate(data):
            h_flag = random.random() > args.alpha
            t_flag = random.random() > args.alpha
            h_p = sentence["h"]["pos"][0] 
            t_p = sentence["t"]["pos"][0]
            ids, ph, pt = entityMarker.tokenize(sentence["tokens"], [h_p[0], h_p[-1]+1], [t_p[0], t_p[-1]+1], None, None, h_flag, t_flag)
            length = min(len(ids), args.max_length)
            self.rel_tokens[i], self.rel_mask[i] = entityMarker.tokenize_rel(sentence, args.max_length)
            self.tokens[i][:length] = ids[:length]
            self.mask[i][:length] = 1
            self.h_pos[i] = min(args.max_length-1, ph) 
            self.t_pos[i] = min(args.max_length-1, pt)
        print("The number of sentence in which tokenizer can't find head/tail entity is %d" % entityMarker.err)


    def __len__(self):
        """Number of instances in an epoch.
        
        Overwitten function.
        """
        return self.len

    def __getitem__(self, index):
        """Get training instance.

        Overwitten function.
        
        Args:
            index: Instance index.
        
        Return:
            input: instances-Tokenized word id.
            mask: Attention mask for instances-bert. 0 means masking, 1 means not masking.
            rel_input: labels-Tokenized word id.
            rel-mask: Attention mask for labels-bert. 0 means masking, 1 means not masking.
            h_pos: Position of head entity.
            t_pos: Position of tail entity.
        """
        input = np.zeros((self.args.max_length), dtype=int)
        mask = np.zeros((self.args.max_length), dtype=int)
        h_pos = np.zeros((1), dtype=int)
        t_pos = np.zeros((1), dtype=int)
        rel_input = np.zeros((self.args.max_length), dtype=int)
        rel_mask = np.zeros((self.args.max_length), dtype=int)

        input = self.tokens[index]
        mask = self.mask[index]
        rel_input = self.rel_tokens[index]
        rel_mask = self.rel_mask[index]
        h_pos = self.h_pos[index]
        t_pos = self.t_pos[index]

        return input, mask, rel_input, rel_mask, h_pos, t_pos

