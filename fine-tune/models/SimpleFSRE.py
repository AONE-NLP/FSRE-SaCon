import sys

sys.path.append('..')
import fewshot_re_kit
import torch
from torch import nn
from torch.nn import functional as F
import math

class SimpleFSRE(fewshot_re_kit.framework.FewShotREModel):

    def __init__(self, sentence_encoder, hidden_size, max_len):
        fewshot_re_kit.framework.FewShotREModel.__init__(self, sentence_encoder)
        self.hidden_size = hidden_size
        self.max_len = max_len
        self.rel_glo_linear = nn.Linear(hidden_size, hidden_size * 2)
        self.ent_glo_linear = nn.Linear(2*hidden_size, hidden_size)
        self.temp_proto = 1  # moco 0.07

    def __dist__(self, x, y, dim):
        return (x * y).sum(dim)

    def __batch_dist__(self, S, Q):
        return self.__dist__(S.unsqueeze(1), Q.unsqueeze(2), 3)

    def forward(self, support, query, rel_text, N, K, total_Q, is_eval=False):
        """
        :param support: Inputs of the support set. (B*N*K)
        :param query: Inputs of the query set. (B*total_Q)
        :param rel_text: Inputs of the relation description.  (B*N)
        :param N: Num of classes
        :param K: Num of instances for each class in the support set
        :param total_Q: Num of instances in the query set
        :param is_eval:
        :return: logits, pred, logits_proto, labels_proto, sim_scalar
        """
        support_ent_glo, support_loc = self.sentence_encoder(support)  # (B * N * K, 2D), (B * N * K, L, D)
        query_ent_glo, query_glo = self.sentence_encoder(query)  # (B * total_Q, 2D), (B * total_Q, L, D)
        rel_text_glo, rel_text_loc = self.sentence_encoder(rel_text, cat=False)  # (B * N, D), (B * N, L, D)
        
        support_ent_glo = support_ent_glo.view(-1, N, K, self.hidden_size*2)
        query_ent_glo = query_ent_glo.view(-1, total_Q, self.hidden_size*2)
        query_glo = query_glo.mean(1).view(-1, total_Q, self.hidden_size)

        
        B = support_ent_glo.shape[0]
       
        rel_text_loc = rel_text_loc.view(B, N, -1, self.hidden_size)
        rel_loc = torch.mean(rel_text_loc, 2) #[B, N, D]


        support_ent_glo = torch.mean(support_ent_glo, 2)
        rel_text_glo = rel_text_glo.view(-1, N, self.hidden_size)
        rel_rep = torch.cat((rel_text_glo, rel_loc), -1)
        rel_rep = rel_rep.view(-1, N, rel_text_glo.shape[2]*2)

        support_proto = support_ent_glo + rel_rep
        query_proto = query_ent_glo

        logits = self.__batch_dist__(support_proto, query_proto)  # (B, total_Q, N)
        minn, _ = logits.min(-1)
        logits = torch.cat([logits, minn.unsqueeze(2) - 1], 2)  # (B, total_Q, N + 1)
        _, pred = torch.max(logits.view(-1, N + 1), 1)

        return logits, pred