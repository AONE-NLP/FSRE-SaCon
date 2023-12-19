import torch
import torch.nn as nn
import numpy as np
from transformers import BertTokenizer, BertModel, BertForSequenceClassification, BertConfig


class BERTSentenceEncoder(nn.Module):

    def __init__(self, pretrain_path, max_length, path1, path2):
        nn.Module.__init__(self)
        self.bert = BertModel.from_pretrained(pretrain_path)
        self.bert2 = BertModel.from_pretrained(pretrain_path)
        # for param in self.bert2.parameters():
        #         param.requires_grad = False
        if path1 is not None and path1 != "None":
            self.bert.load_state_dict(torch.load(path1)["bert-base"], False)
            print("We load " + path1 + " to train!")
        if path2 is not None and path2 != "None":
            self.bert2.load_state_dict(torch.load(path2)["bert-base"], False)
            print("We load " + path2 + " to train!")
        self.max_length = max_length
        self.max_length_name = 8
        self.tokenizer = BertTokenizer.from_pretrained(pretrain_path)

    def forward(self, inputs, cat=True, flag=0):
        # outputs = self.bert(inputs['word'], attention_mask=inputs['mask'])

        if flag == 1:
            outputs = self.bert(inputs['word'], attention_mask=inputs['mask'])
            tensor_range = torch.arange(inputs['word'].size()[0])
            h_state = outputs[0][tensor_range, inputs["pos1"]]
            t_state = outputs[0][tensor_range, inputs["pos2"]]
            state = torch.cat((h_state, t_state), -1)

            return state

        elif flag == 2:
            outputs = self.bert(inputs['word'], attention_mask=inputs['mask'])
            # xs, x = self.bert(inputs['word'], attention_mask=inputs['mask'])
            # print(xs)
            # print(x)
            return outputs[0] 
        else:
            if cat:
                outputs = self.bert(inputs['word'], attention_mask=inputs['mask'])
                tensor_range = torch.arange(inputs['word'].size()[0])
                # mask = torch.zeros(tensor_range.shape[0], dtype=torch.bool)
                # mask = inputs['ent_mask']
                # mask[inputs["pos1"]:inputs["pos2"]] = 1
                h_state = outputs[0][tensor_range, inputs["pos1"]]
                t_state = outputs[0][tensor_range, inputs["pos2"]]
                state = torch.cat((h_state, t_state), -1)
                # max_state = (mask.unsqueeze(-1)*outputs[0]).max(dim=1)[0]
                return state, outputs[0]
            else:
                outputs = self.bert2(inputs['word'], attention_mask=inputs['mask'])
                return outputs[1], outputs[0]

    def tokenize(self, raw_tokens, pos_head, pos_tail):
        # token -> index
        tokens = ['[CLS]']
        cur_pos = 0
        pos1_in_index = 0
        pos2_in_index = 0
        for token in raw_tokens:
            token = token.lower()
            if cur_pos == pos_head[0]:
                tokens.append('[unused0]')
                pos1_in_index = len(tokens)
            if cur_pos == pos_tail[0]:
                tokens.append('[unused1]')
                pos2_in_index = len(tokens)
            tokens += self.tokenizer.tokenize(token)
            if cur_pos == pos_head[-1]:
                tokens.append('[unused2]')
            if cur_pos == pos_tail[-1]:
                tokens.append('[unused3]')
            cur_pos += 1
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)
        
        # padding
        while len(indexed_tokens) < self.max_length:
            indexed_tokens.append(0)
        indexed_tokens = indexed_tokens[:self.max_length]

    
        # pos
        pos1 = np.zeros(self.max_length, dtype=np.int32)
        pos2 = np.zeros(self.max_length, dtype=np.int32)
        for i in range(self.max_length):
            pos1[i] = i - pos1_in_index + self.max_length
            pos2[i] = i - pos2_in_index + self.max_length
    
        # mask
        mask = np.zeros(self.max_length, dtype=np.int32)
        mask[:len(tokens)] = 1

        # ent_mask = np.zeros(self.max_length, dtype=np.int32)
        # if pos_head[-1] < pos_tail[0]:
        #     ent_mask[pos_head[-1]:pos_tail[0]] = 1
        # else: 
        #     ent_mask[pos_tail[-1]:pos_head[0]] = 1

        # ent_mask[pos_head] = 1
        # ent_mask[pos_tail] = 1
        
        if pos1_in_index == 0:
            pos1_in_index = 1
        if pos2_in_index == 0:
            pos2_in_index = 1
        pos1_in_index = min(self.max_length, pos1_in_index)
        pos2_in_index = min(self.max_length, pos2_in_index)
    
        return indexed_tokens, pos1_in_index - 1, pos2_in_index - 1, mask

    def tokenize_reg(self, raw_tokens, pos_head, pos_tail):
        # token -> index
        tokens = ['[CLS]']
        cur_pos = 0
        pos1_in_index = 0
        pos2_in_index = 0
        for token in raw_tokens:
            token = token.lower()
            if cur_pos == pos_head[0]:
                tokens.append('[unused0]')
                pos1_in_index = len(tokens)
            if cur_pos == pos_tail[0]:
                tokens.append('[unused1]')
                pos2_in_index = len(tokens)
            tokens += self.tokenizer.tokenize(token)
            if cur_pos == pos_head[-1]:
                tokens.append('[unused2]')
            if cur_pos == pos_tail[-1]:
                tokens.append('[unused3]')
            cur_pos += 1
        pos = []
        for k, token in enumerate(tokens):
            if token == '[unused0]':
                pos += [min(k, self.max_length - 1)]
            if token == '[unused2]':
                pos += [min(k, self.max_length - 1)]
        for k, token in enumerate(tokens):
            if token == '[unused1]':
                pos += [min(k, self.max_length - 1)]
            if token == '[unused3]':
                pos += [min(k, self.max_length - 1)]
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)
        
        # padding
        while len(indexed_tokens) < self.max_length:
            indexed_tokens.append(0)
        indexed_tokens = indexed_tokens[:self.max_length]

        # pos
        #pos1 = np.zeros((self.max_length), dtype=np.int32)
        #pos2 = np.zeros((self.max_length), dtype=np.int32)
        #for i in range(self.max_length):
        #    pos1[i] = i - pos1_in_index + self.max_length
        #    pos2[i] = i - pos2_in_index + self.max_length

        # mask
        mask = np.zeros((self.max_length), dtype=np.int32)
        mask[:len(tokens)] = 1

        return indexed_tokens, pos, mask

    def tokenize_rel(self, raw_tokens):
        # token -> index
        tokens = ['[CLS]']
        name, description = raw_tokens
        for token in name.split(' '):
            token = token.lower()
            tokens += self.tokenizer.tokenize(token)
        tokens.append('[SEP]')
        for token in description.split(' '):
            token = token.lower()
            tokens += self.tokenizer.tokenize(token)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)

        # padding
        while len(indexed_tokens) < self.max_length:
            indexed_tokens.append(0)
        indexed_tokens = indexed_tokens[:self.max_length]

        # mask
        mask = np.zeros(self.max_length, dtype=np.int32)
        mask[:len(tokens)] = 1

        return indexed_tokens, mask

    def tokenize_name(self, name):
        # for FewRel 2.0
        # token -> index
        tokens = ['[CLS]']
        for token in name.split('_'):
            token = token.lower()
            tokens += self.tokenizer.tokenize(token)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)

        # padding
        while len(indexed_tokens) < self.max_length_name:
            indexed_tokens.append(0)
        indexed_tokens = indexed_tokens[:self.max_length_name]

        # mask
        mask = np.zeros(self.max_length_name, dtype=np.int32)
        mask[:len(tokens)] = 1

        return indexed_tokens, mask

class BERTPAIRSentenceEncoder(nn.Module):

    def __init__(self, pretrain_path, max_length, path1): 
        nn.Module.__init__(self)
        self.bert = BertForSequenceClassification.from_pretrained(
                pretrain_path,
                num_labels=2)

        if path1 is not None and path1 != "None":
            self.bert.bert.load_state_dict(torch.load(path1)["bert-base"], False)
            print("We load " + path1 + " to train!")

        
        self.max_length = max_length
        self.tokenizer = BertTokenizer.from_pretrained(pretrain_path)
        ##
        #self.bilstm = BiLSTM(768)
    
    
    def windows_sequence(self,sequence_output, windows, lstm_layer):
        batch_size, max_len, feat_dim = sequence_output.shape
        local_final = torch.zeros([batch_size, max_len, feat_dim], dtype=torch.float32, device='cuda')
        for i in range(max_len):
            index_list = []
            for u in range(1, windows // 2 + 1):
                if i - u >= 0:
                    index_list.append(i - u)
                if i + u <= max_len - 1:
                    index_list.append(i + u)
            index_list.append(i)
            index_list.sort()
            temp = sequence_output[:, index_list, :]
            out,(h,b) = lstm_layer(temp)
            local_f = out[:, -1, :]
            local_final[:, i, :] = local_f
        return local_final
    
    def forward(self, inputs):
        x = self.bert(inputs['word'], token_type_ids=inputs['seg'], attention_mask=inputs['mask'])[0] ## x.shape [100, 2]
        
        #import pdb
        #pdb.set_trace()
        ##
        #local_x = self.windows_sequence(x, 5, self.bilstm)
        #x = torch.cat((x, local_x), -1)
        
        return x
    
    def tokenize(self, raw_tokens, pos_head, pos_tail):
        # token -> index
        # tokens = ['[CLS]']
        tokens = []
        cur_pos = 0
        pos1_in_index = 0
        pos2_in_index = 0
        for token in raw_tokens:
            token = token.lower()
            if cur_pos == pos_head[0]:
                tokens.append('[unused0]')
                pos1_in_index = len(tokens)
            if cur_pos == pos_tail[0]:
                tokens.append('[unused1]')
                pos2_in_index = len(tokens)
            tokens += self.tokenizer.tokenize(token)
            if cur_pos == pos_head[-1]:
                tokens.append('[unused2]')
            if cur_pos == pos_tail[-1]:
                tokens.append('[unused3]')
            cur_pos += 1
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)
        
        return indexed_tokens
