import os
from re import T

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from fewshot_re_kit.data_loader import get_loader
from fewshot_re_kit.data_loader import get_loader_lpd
from fewshot_re_kit.data_loader import get_loader22
from fewshot_re_kit.data_loader import get_loader_pair
from fewshot_re_kit.data_loader import get_loader_reg
from fewshot_re_kit.data_loader import get_loader_reg2
from fewshot_re_kit.data_loader import get_loader_pair2
from fewshot_re_kit.data_loader import get_loader_test_lpd
from fewshot_re_kit.framework import FewShotREFramework
from fewshot_re_kit.sentence_encoder import BERTSentenceEncoder, BERTPAIRSentenceEncoder
import models
from models.SimpleFSRE import SimpleFSRE
from models.HCRP import HCRP
from models.LPD import LPD
from models.PAIR import PAIR
from models.GNN import GNN
from models.Proto import Proto
from models.REGRAB import REGRAB
import sys
import torch
from torch import optim, nn
import numpy as np
import json
import argparse
import torch
import random
import time
import pickle

def setup_seed(seed):
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed(seed)
		torch.cuda.manual_seed_all(seed)
	random.seed(seed)
	np.random.seed(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmard = False
	torch.random.manual_seed(seed)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='./data',
                        help='file root')
    parser.add_argument('--train', default='train_wiki',
                        help='train file')
    parser.add_argument('--val', default='val_wiki',
                        help='val file')
    parser.add_argument('--test', default='test_wiki5-1',
                        help='test file')
    parser.add_argument('--ispubmed', default=False, type=bool,
                       help='FewRel 2.0 or not')
    parser.add_argument('--pid2name', default='pid2name',
                        help='pid2name file: relation names and description')
    parser.add_argument('--trainN', default=10, type=int,
                        help='N in train')
    parser.add_argument('--N', default=10, type=int,
                        help='N way')
    parser.add_argument('--K', default=1, type=int,
                        help='K shot')
    parser.add_argument('--Q', default=1, type=int,
                        help='Num of query per class')
    parser.add_argument('--batch_size', default=4, type=int,
                        help='batch size')
    parser.add_argument('--train_iter', default=30000, type=int,
                        help='num of iters in training')
    parser.add_argument('--val_iter', default=1000, type=int,
                        help='num of iters in validation')
    parser.add_argument('--test_iter', default=10000, type=int,
                        help='num of iters in testing')
    parser.add_argument('--val_step', default=1000, type=int,
                        help='val after training how many iters')
    parser.add_argument('--model', default=None,
                        help='model name')
    parser.add_argument('--encoder', default='bert',
                        help='encoder: bert')
    parser.add_argument('--max_length', default=128, type=int,
                        help='max length')
    parser.add_argument('--lr', default=2e-5, type=float,
                        help='learning rate')
    parser.add_argument('--weight_decay', default=0.01, type=float,
                        help='weight decay')
    parser.add_argument('--lamda', default=1, type=float,
                        help='loss combination')
    parser.add_argument('--grad_iter', default=1, type=int,
                        help='accumulate gradient every x iterations')
    parser.add_argument('--optim', default='adamw',
                        help='sgd / adam / adamw')
    parser.add_argument('--hidden_size', default=768, type=int,
                        help='hidden size')
    parser.add_argument('--label_mask_prob', default=0.4, type=float)
    parser.add_argument('--load_ckpt', default=None,
                        help='load ckpt')
    parser.add_argument('--save_ckpt', default=None,
                        help='save ckpt')
    parser.add_argument('--only_test', default=False,
                        help='only test')
    parser.add_argument('--pretrain_ckpt', default='bert-base-uncased',
                        help='bert / roberta pre-trained checkpoint')
    parser.add_argument('--seed', default=42, type=int,
                        help='seed') #19961228
    parser.add_argument('--path1', default='ckpt/SaCon1',
                        help='path to ckpt')
    parser.add_argument('--path2', default='ckpt/SaCon2',
    help='path to ckpt')
    parser.add_argument('--na_rate', default=1, type=int,
        help='NA rate (NA = Q * na_rate)')
    parser.add_argument('--pair', action='store_true', default=False,
           help='use pair model')
    parser.add_argument("--local_rank", dest="local_rank", type=int,
                        default=-1, help="local rank")
    parser.add_argument('--eps', default=0.1, type=float,
        help='step size for SG-MCMC')
    parser.add_argument('--temp', default=10.0, type=float,
        help='temperature for softmax')
    parser.add_argument('--step', default=5, type=int,
           help='steps for SG-MCMC')
    parser.add_argument('--smp', default=10, type=int,
           help='samples for SG-MCMC')
    parser.add_argument('--ratio', default=0.01, type=float,
           help='decay ratio of step size for SG-MCMC')
    parser.add_argument('--wtp', default=0.1, type=float,
           help='weight of the prior term')
    parser.add_argument('--wtn', default=1.0, type=float,
           help='weight of the noise term')
    parser.add_argument('--wtb', default=0.0, type=float,
           help='weight of the background term')
    parser.add_argument('--metric', default='dot',
           help='similarity metric (dot or l2)')

    opt = parser.parse_args()
    trainN = opt.trainN
    N = opt.N
    K = opt.K
    Q = opt.Q
    batch_size = opt.batch_size
    model_name = opt.model
    encoder_name = opt.encoder
    max_length = opt.max_length

    only_test = opt.only_test

    print("{}-way-{}-shot Few-Shot Relation Classification".format(N, K))
    print("model: {}".format(model_name))
    print("encoder: {}".format(encoder_name))
    print("max_length: {}".format(max_length))

    setup_seed(opt.seed)

    # encoder
    if opt.pair:
        sentence_encoder = BERTPAIRSentenceEncoder(
                opt.pretrain_ckpt,
                max_length,
                path1=opt.path1)
    else:
        sentence_encoder = BERTSentenceEncoder(opt.pretrain_ckpt, max_length, path1=opt.path1, path2=opt.path2)

    # train / val / test data loader
    if model_name == 'SimpleFSRE' or model_name == 'HCRP' or model_name == 'GNN' or model_name == 'Proto':
        train_data_loader = get_loader(opt.train, opt.pid2name, sentence_encoder,
                                    N=trainN, K=K, Q=Q, batch_size=batch_size, ispubmed=opt.ispubmed, root=opt.root)
        val_data_loader = get_loader(opt.val, opt.pid2name, sentence_encoder,
                                    N=N, K=K, Q=Q, batch_size=batch_size, ispubmed=opt.ispubmed, root=opt.root)
        test_data_loader = get_loader22(opt.test, opt.pid2name, sentence_encoder,
                                    N=N, K=K, Q=Q, batch_size=batch_size, ispubmed=opt.ispubmed, root=opt.root)
    elif model_name == 'LPD':
        train_data_loader = get_loader_lpd(opt, opt.train, sentence_encoder,
                N=trainN, K=K, Q=Q,  batch_size=batch_size, label_mask_prob=opt.label_mask_prob)
        val_data_loader = get_loader_lpd(opt, opt.val, sentence_encoder,
                    N=N, K=K, Q=Q,  batch_size=batch_size, label_mask_prob=0.0)
        test_data_loader = get_loader_test_lpd(opt.test, sentence_encoder, batch_size=batch_size)
    elif model_name == 'PAIR':
        train_data_loader = get_loader_pair(opt.train, sentence_encoder,
                N=trainN, K=K, Q=Q, na_rate=opt.na_rate, batch_size=batch_size, encoder_name=encoder_name)
        val_data_loader = get_loader_pair(opt.val, sentence_encoder,
                N=N, K=K, Q=Q, na_rate=opt.na_rate, batch_size=batch_size, encoder_name=encoder_name)
        test_data_loader = get_loader_pair2(opt.test, sentence_encoder,
                N=N, K=K, Q=Q, na_rate=opt.na_rate, batch_size=batch_size, encoder_name=encoder_name)
    elif model_name == 'REGRAB':
        train_data_loader = get_loader_reg(opt.train, sentence_encoder,
                N=trainN, K=K, Q=Q, na_rate=opt.na_rate, batch_size=batch_size)
        val_data_loader = get_loader_reg(opt.val, sentence_encoder,
                N=N, K=K, Q=Q, na_rate=opt.na_rate, batch_size=batch_size)
        test_data_loader = get_loader_reg2(opt.test, sentence_encoder,
                N=N, K=K, Q=Q, batch_size=batch_size)

    framework = FewShotREFramework(train_data_loader, val_data_loader, test_data_loader)

    if model_name == 'SimpleFSRE':
        model = SimpleFSRE(sentence_encoder, hidden_size=opt.hidden_size, max_len=max_length)
    elif model_name == 'HCRP':
        model = HCRP(sentence_encoder, hidden_size=opt.hidden_size, max_len=max_length)
    elif model_name == 'LPD':
        model = LPD(sentence_encoder, hidden_size=opt.hidden_size)
    elif model_name == 'PAIR':
        model = PAIR(sentence_encoder, hidden_size=opt.hidden_size)
    elif model_name == 'GNN':
        model = GNN(sentence_encoder, N, hidden_size=opt.hidden_size)
    elif model_name == 'Proto':
        model = Proto(sentence_encoder, hidden_size=opt.hidden_size)
    elif model_name == 'REGRAB':
        # Loading relation embeddings.
        data = pickle.load(open('./data/embeddings.pkl', 'rb'))
        rellist = data['relations']
        relemb = data['embeddings']
        array0 = np.zeros((1,relemb.shape[1]), dtype=relemb.dtype)
        relemb = np.concatenate([array0, relemb], axis=0)
        rel2id = dict([(rel, k + 1) for k, rel in enumerate(rellist)])
        # Loading relation graphs.
        with open('./data/graph.txt', 'r') as fi:
            us, vs, ws = [], [], []
            for line in fi:
                items = line.strip().split('\t')
                us += [rel2id[items[0]]]
                vs += [rel2id[items[1]]]
                ws += [float(items[2])]
            index = torch.LongTensor([us, vs])
            value = torch.Tensor(ws)
            shape = torch.Size([len(rel2id) + 1, len(rel2id) + 1])
            reladj = torch.sparse.FloatTensor(index, value, shape).cuda()
        model = REGRAB(sentence_encoder, hidden_size=opt.hidden_size, eps=opt.eps, temp=opt.temp, step=opt.step, smp=opt.smp, ratio=opt.ratio, wtp=opt.wtp, wtn=opt.wtn, wtb=opt.wtb, metric=opt.metric)
        model.set_relemb(rel2id, relemb)
        model.set_reladj(reladj)

    if torch.cuda.is_available():
        model.cuda()

    # model save path
    if not os.path.exists('checkpoint'):
        os.mkdir('checkpoint')
    prefix = '-'.join([model_name, encoder_name, opt.train, opt.val, str(N), str(K)])
    ckpt = 'checkpoint/{}.pth.tar'.format(prefix)
    if opt.save_ckpt:
        ckpt = opt.save_ckpt

    if not opt.only_test:
        T1 = time.clock()
        framework.train(model, model_name, prefix, trainN, N, K, Q, learning_rate=opt.lr, weight_decay=opt.weight_decay,
                        lamda=opt.lamda, train_iter=opt.train_iter, val_iter=opt.val_iter,
                        load_ckpt=opt.load_ckpt, save_ckpt=ckpt, val_step=opt.val_step, grad_iter=opt.grad_iter, na_rate=opt.na_rate)
        T2 = time.clock()
        print('total training time:%s s' % (T2 - T1))

        acc = framework.eval(model, model_name, N, K, Q, opt.test_iter, ckpt=ckpt, test=only_test, na_rate=opt.na_rate)

    else:
        ckpt = opt.load_ckpt
        acc = framework.eval(model, model_name, N, K, Q, opt.test_iter, ckpt=ckpt, test=only_test, na_rate = opt.na_rate)

    T3 = time.clock()
    T4 = time.clock()
    print('total evaluation time:%s s' % (T4 - T3))
    print("RESULT: %.2f" % (acc * 100))


if __name__ == "__main__":
    main()
