import torch
import json
import re
import os
from os.path import join
import argparse
from model.data import SampleDataset
from model.bert_model import BertMatcher
from torch.utils.data import DataLoader
from model.batcher import coll_fn
from pytorch_pretrained_bert import BertTokenizer
from model.batcher import pad_batch_tensorize
from toolz.sandbox import unzip
try:
    DATA_DIR = 'matcher'
    DATASET_DIR='data/class/'
except KeyError:
    print('please use environment variable to specify data directories')

class MatcherDataset(SampleDataset):
    def __init__(self,split):
        super().__init__(split,DATASET_DIR)

    def __getitem__(self, i):
        js_data = super().__getitem__(i)

        label , question , text = (js_data['is_related'],js_data['question'],js_data['related_context'])
        concat_text="[CLS] "+ question+ " [SEP]"+ text+ " [SEP]"
        return label, concat_text
def load_best_ckpt(model_dir, reverse=False):

    ckpts = os.listdir(join(model_dir, 'ckpt'))
    ckpt_matcher = re.compile('^ckpt-.*-[0-9]*')
    ckpts = sorted([c for c in ckpts if ckpt_matcher.match(c)],
                   key=lambda c: float(c.split('-')[1]), reverse=reverse)
    print('loading checkpoint {}...'.format(ckpts[0]))
    ckpt = torch.load(
        join(model_dir, 'ckpt/{}'.format(ckpts[0]))
    )['state_dict']
    return ckpt

def tokenize(max_pos,concat_texts):
    tokenizer = BertTokenizer.from_pretrained('../ERINE_pretrain')
    tokenized_texts=[tokenizer.tokenize(concat_text) for concat_text in concat_texts]
    indexed_tokens = [tokenizer.convert_tokens_to_ids(tokenized_text) for tokenized_text in tokenized_texts]
    indexed_tokens=[indexed_token[:max_pos] for indexed_token in indexed_tokens]

    return indexed_tokens

def prepro_fn(max_pos, batch):

    labels, concat_texts = batch

    indexed_tokens=tokenize(max_pos, concat_texts)
    batch=list(zip(labels,indexed_tokens))

    return batch


def main(args):
    meta = json.load(open(join(DATA_DIR, 'meta.json')))
    nargs = meta['net_args']
    ckpt = load_best_ckpt(DATA_DIR)
    net=BertMatcher(**nargs)
    net.load_state_dict(ckpt)
    if args.cuda:
        net = net.cuda()
    net.eval()
    loader = DataLoader(MatcherDataset('test'), batch_size=1,
                        num_workers=4 if args.cuda else 0, collate_fn=coll_fn)
    tokenizer = BertTokenizer.from_pretrained('./MRC_pretrain')
    count=0
    
    with torch.no_grad():
        for index in range(800):

            with open(join(join(DATASET_DIR,'test'),'{}.json'.format(index+1))) as f:
                js_data = json.load(f)
                print('loading: {}'.format(index+1))
                label, question, text = (js_data['is_related'], js_data['question'], js_data['text'])
                concat_text =question +text

           
            # tokenized_text = tokenizer.tokenize(concat_text)
            # indexed_token = tokenizer.convert_tokens_to_ids(tokenized_text)
            # indexed_token = [indexed_token[:512]]

            token_tensor, segment_tensor, mask_tensor = pad_batch_tensorize([concat_text], args.cuda)

            fw_args = (token_tensor, segment_tensor, mask_tensor)
            net_out = net(*fw_args)

            print('label: {}'.format(label))
            print('net_out: {}'.format( net_out[0][0].item()))
            i=1 if net_out[0][0].item()>0.9 else 0
            if(i==label):
                count+=1
    print('accuracy: {} / 800'.format(count))





if __name__ == '__main__':
    print(torch.cuda.is_available())
    parser = argparse.ArgumentParser(
        description='trainingtest of bert matcher'
    )
    parser.add_argument('--no-cuda', action='store_true',
                        help='disable GPU training')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available() and not args.no_cuda

    main(args)