import os 
import pdb 
import torch
import torch.nn as nn 
from pytorch_metric_learning.losses import NTXentLoss
from transformers import BertForMaskedLM, BertForPreTraining, BertTokenizer
from transformers import BertTokenizer, BertModel
import numpy as np 

def mask_tokens(inputs, tokenizer, not_mask_pos=None):
    """Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
    
    Args:
        inputs: Inputs to mask. (batch_size, max_length) 
        tokenizer: Tokenizer.
        not_mask_pos: Using to forbid masking entity mentions. 1 for not mask.
    
    Returns:
        inputs: Masked inputs.
        labels: Masked language model labels.
    """

    if tokenizer.mask_token is None:
        raise ValueError(
            "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
        )

    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, 0.15)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
    ]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    if tokenizer._pad_token is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)
    if not_mask_pos is None:
        masked_indices = torch.bernoulli(probability_matrix).bool()
    else:
        masked_indices = torch.bernoulli(probability_matrix).bool() & (~(not_mask_pos.bool())) # ** can't mask entity marker **
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs.cuda(), labels.cuda()

'''
contrastive loss
'''
class one_hot_CrossEntropy(torch.nn.Module):
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def __init__(self):
        super(one_hot_CrossEntropy,self).__init__()
    def forward(self, x, y):
        P_i = torch.nn.functional.softmax(x, dim=1)
        loss = y*torch.log(P_i + 0.0000001)
        loss = -torch.mean(torch.sum(loss, dim=1), dim=0)
        return loss

class SaCon(nn.Module):
    """Contrastive Pre-training model.

    This class implements `SaCon` model based on model `BertForMaskedLM`.

    Attributes:
        model: Model to train.
        tokenizer: Tokenizer.
        args: Args from command line. 
    """
    def __init__(self, args):
        super(SaCon, self).__init__()
        self.model = BertForMaskedLM.from_pretrained('bert-base-uncased')
        self.model2 = BertForMaskedLM.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.ntxloss = NTXentLoss(temperature=args.temperature)
        self.onehotloss1 = one_hot_CrossEntropy()
        self.onehotloss2 = one_hot_CrossEntropy()
        self.args = args 
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
    
    def forward(self, input, mask, rel_input, rel_mask, h_pos, t_pos):

        input = input.type(torch.long)
        h_pos = h_pos.type(torch.long)
        t_pos = t_pos.type(torch.long)

        #filter duple relation
        rel_filter = torch.unique(rel_input, dim=0, return_inverse=True)
        new_index = rel_filter[1]
        new_index_change = torch.where(new_index==0, -2, new_index)
        y, indices = new_index_change.sort()
        y[1:] *= ((y[1:] - y[:-1]) !=0).long()
        indices = indices.sort(dim=-1)[1]
        result = torch.gather(y, 0, indices)
        x = torch.where(result==0, -1, result)
        z = torch.where(x==-2, 0, x)

        a = torch.nonzero(z==-1).squeeze(1)

        del_re = torch.arange(rel_input.size(0)).apply_(lambda l: l not in a).bool()
        rel_input_new = rel_input[del_re, :]

        u = torch.index_select(rel_input, dim=0, index=a)
        v = rel_input_new
        matching_rows = torch.all(torch.unsqueeze(u, 1) == v, dim=2)
        new_indices = torch.nonzero(matching_rows)
        A = new_indices[:,1]



        pos = torch.eye(rel_input.shape[0], dtype=int)
        del_pos = torch.arange(pos.size(1)).apply_(lambda l: l not in a).bool()
        final_pos = pos[:, del_pos]

        B = torch.cat((a.unsqueeze(1), A.unsqueeze(1)), 1)
        P =torch.ones(B.shape[0], dtype=int)
        final_pos = final_pos.index_put_(tuple(B.t()), P)

        del_mask = torch.arange(rel_mask.size(0)).apply_(lambda l: l not in a).bool()
        rel_mask = rel_mask[del_mask,:]

        rel_input = v.type(torch.long)


        # Ensure that `mask_tokens` function doesn't mask entity mention.
        indice = torch.arange(0, input.size()[0])
        not_mask_pos = torch.zeros((input.size()[0], input.size()[1]), dtype=int)
        not_mask_pos[indice, h_pos] = 1
        not_mask_pos[indice, t_pos] = 1


        m_input, m_labels = mask_tokens(input.cpu(), self.tokenizer, not_mask_pos)
        m_rel_input, m_rel_labels = mask_tokens(rel_input.cpu(), self.tokenizer)
        m_outputs = self.model(input_ids=m_input, labels=m_labels, attention_mask=mask, output_hidden_states=True)
        outputs = m_outputs[2][-1]
        m_loss = m_outputs[0]
        m_rel_outputs = self.model2(input_ids=m_rel_input, labels=m_rel_labels, attention_mask=rel_mask, output_hidden_states=True)
        m_rel_loss = m_rel_outputs[0]
        rel_outputs = m_rel_outputs[2][-1]

        # entity marker starter
        batch_size = input.size()[0]
        batch_rel = rel_mask.shape[0]
        indice = torch.arange(0, batch_size)
        rel_indice = torch.arange(0, batch_rel)
        h_state = outputs[indice, h_pos] # (batch_size * 2, hidden_size)
        t_state = outputs[indice, t_pos]

        cls_pos = torch.zeros(batch_rel).type(torch.long)
        r_state_global = rel_outputs[rel_indice, cls_pos]
        r_state_local = torch.mean(rel_outputs, 1)
        r_state = torch.cat((r_state_global, r_state_local), 1)
        e_state = torch.cat((h_state, t_state), 1)

        r_state = r_state / r_state.norm(dim=1, keepdim=True)
        e_state = e_state / e_state.norm(dim=1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits_per_entity = logit_scale * e_state @ r_state.t()
        logits_per_relation = logits_per_entity.t()

        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        final_pos = final_pos.to(_device)

        loss_entity = self.onehotloss1(logits_per_entity, final_pos)
        loss_relation = self.onehotloss2(logits_per_relation, final_pos.t())


        r_loss = (loss_entity + loss_relation) / 2


        return m_loss+m_rel_loss, r_loss