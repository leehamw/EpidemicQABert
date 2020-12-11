import argparse
import json
import os
from os.path import join, exists
import pickle as pkl
from pytorch_pretrained_bert import BertTokenizer
import math
import torch
from torch import optim
import torch.nn as nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from model.data import SampleDataset
from model.batcher import prepro_fn, batchify_fn,coll_fn
from model.batcher import BucketedGenerater

from model.bert_model import BertMatcher

from model.training import get_basic_grad_fn, basic_validate
from model.training import BasicPipeline, BasicTrainer


try:
    DATA_DIR='data/class/'

except KeyError:
    print('please use environment variable to specify data directories')

class MatchDataset(SampleDataset):
    def __init__(self,split):
        super().__init__(split,DATA_DIR)

    def __getitem__(self, i):
        js_data = super().__getitem__(i)

        label , question , text = (js_data['is_related'],js_data['question'],js_data['text'])
        concat_text=question + text
        return label, concat_text

#finetune, hidden_dim, max_pos
def configure_net(finetune, hidden_dim, max_pos):
    net_args = {}
    net_args['finetune'] = finetune
    net_args['hidden_dim'] = hidden_dim
    net_args['max_pos'] = max_pos

    net=BertMatcher(**net_args)

    return net, net_args

def configure_training(opt, lr, clip_grad, lr_decay, batch_size):
    """ supports Adam optimizer only"""
    assert opt in ['adam']
    opt_kwargs = {}
    opt_kwargs['lr'] = lr

    train_params = {}
    train_params['optimizer']      = (opt, opt_kwargs)
    train_params['clip_grad_norm'] = clip_grad
    train_params['batch_size']     = batch_size
    train_params['lr_decay']       = lr_decay



    def criterion(logits, targets):
        k = torch.nn.BCELoss()
        logits=torch.squeeze(logits,dim=-1)
        loss=k(logits, targets)
        # print(loss)
        # print(type(loss))
        assert (not math.isnan(loss.item())
                and not math.isinf(loss.item()))

        return loss
    return criterion, train_params

def build_batchers(max_pos, cuda, debug):
    prepro = prepro_fn(max_pos)

    def sort_key(sample):
        label, indexed_tokens=sample
        return len(indexed_tokens)

    batchify=batchify_fn(cuda=cuda)
    train_loader = DataLoader(MatchDataset('train'),batch_size=11000,shuffle=not debug,
                              num_workers=4 if cuda and not debug else 0, collate_fn=coll_fn)

    train_batcher= BucketedGenerater(train_loader, sort_key, prepro, batchify, single_run=False,
                                     fork=not debug)

    val_loader = DataLoader(MatchDataset('valid'),batch_size=2000,shuffle=False,
                            num_workers=4 if cuda and not debug else 0, collate_fn=coll_fn)

    val_batcher=BucketedGenerater(val_loader, sort_key, prepro, batchify, single_run=True,
                                  fork=not debug)

    return train_batcher, val_batcher

def main(args):
    net, net_args = configure_net(args.finetune, args.hidden_dim, args.max_pos)

    train_batcher, val_batcher=build_batchers(args.max_pos, args.cuda, args.debug)

    criterion, train_params = configure_training(
        'adam', args.lr, args.clip, args.decay, args.batch
    )

    if not exists(args.path):
        os.makedirs(args.path)
    meta = {}
    meta['net'] = 'bert_matcher'
    meta['net_args'] = net_args
    meta['traing_params'] = train_params
    with open(join(args.path, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=4)

    # prepare trainer
    val_fn = basic_validate(net, criterion)
    grad_fn = get_basic_grad_fn(net, args.clip)
    optimizer = optim.Adam(net.parameters(), **train_params['optimizer'][1]) #一个星(*):表示接收的参数作为元组来处理两个星(**):表示接收的参数作为字典来
    scheduler = ReduceLROnPlateau(optimizer, 'min', verbose=True,
                                  factor=args.decay, min_lr=0,
                                  patience=args.lr_p)
    if args.cuda:
        net = net.cuda()
    pipeline = BasicPipeline(meta['net'], net,
                             train_batcher, val_batcher, args.batch, val_fn,
                             criterion, optimizer, grad_fn)
    trainer = BasicTrainer(pipeline, args.path,
                           args.ckpt_freq, args.patience, scheduler)

    print('start training with the following hyper-parameters:')
    print(meta)
    trainer.train()


if __name__ == '__main__':
    print('6400')
    print(torch.cuda.is_available())
    parser = argparse.ArgumentParser(
        description='training of bert matcher'
    )
    parser.add_argument('--path', help='root of the model', default='matcher')
    parser.add_argument('--max_pos', help='max_pos of bert', default=512)
    parser.add_argument('--hidden_dim', help='hidden_dim of linear layer', default=1024)
    parser.add_argument('--lr', type=float, action='store', default=2e-5,
                        help='learning rate')
    parser.add_argument('--decay', type=float, action='store', default=0.5,
                        help='learning rate decay ratio')
    parser.add_argument('--lr_p', type=int, action='store', default=0,
                        help='patience for learning rate decay')
    parser.add_argument('--clip', type=float, action='store', default=2.0,
                        help='gradient clipping')
    parser.add_argument('--batch', type=int, action='store', default=20,
                        help='the training batch size')
    parser.add_argument('--patience', type=int, action='store', default=5,
                        help='patience for early stopping')
    parser.add_argument(
        '--ckpt_freq', type=int, action='store', default=180,
        help='number of update steps for checkpoint and validation'
    )
    parser.add_argument('--debug', action='store_true',
                        help='run in debugging mode')
    parser.add_argument('--finetune', action='store_false',
                        help='finetune in bert model')
    parser.add_argument('--no-cuda', action='store_true',
                        help='disable GPU training')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available() and not args.no_cuda

    main(args)

