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
    DATASET_DIR = 'data/class/'
    CONTEXT_DIR='data/class/context'
    AFTER_DIR='data/final/test_sample_with_context'
except KeyError:
    print('please use environment variable to specify data directories')





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





def main(args):
    print('./MRC_pretrain')
    os.makedirs(AFTER_DIR)
    meta = json.load(open(join(DATA_DIR, 'meta.json')))
    nargs = meta['net_args']
    ckpt = load_best_ckpt(DATA_DIR)
    net = BertMatcher(**nargs)
    net.load_state_dict(ckpt)
    if args.cuda:
        net = net.cuda()
    net.eval()
    tokenizer = BertTokenizer.from_pretrained('./MRC_pretrain')


    with torch.no_grad():
        for index in range(143):


            with open(join(join('data/final', 'original_test_sample'), '{}.json'.format(index + 1501))) as f:
                js_data = json.load(f)
                print('loading: {}'.format(index + 1501))
                id, question_text, ques_id = (js_data['id'], js_data['question'], js_data['question_id'])
            highest_score=[0]
            context_new_id=[]
            context_id=[]
            context_content=[]
            for con in range(8932):

                with open(join(join(DATASET_DIR, 'context'), '{}.json'.format(con + 1))) as c:
                    cn_data = json.load(c)
                    new_docid, docid, text=(cn_data['new_docid'], cn_data['docid'], cn_data['text'])

                text_tok = tokenizer.tokenize(text)
                text_id = tokenizer.convert_tokens_to_ids(text_tok)
                text_len = len(text_id)

                question_len = len(ques_id)
                if (question_len + text_len <= 512):
                    concat_text=ques_id+text_id


                    token_tensor, segment_tensor, mask_tensor = pad_batch_tensorize([concat_text], args.cuda)

                    fw_args = (token_tensor, segment_tensor, mask_tensor)
                    net_out = net(*fw_args)

                    if (net_out[0][0].item() > highest_score[-1]) :
                        highest_score.clear()
                        highest_score.append(net_out[0][0].item())
                        context_new_id.clear()
                        context_new_id.append(new_docid)
                        context_id.clear()
                        context_id.append(docid)
                        context_content.clear()
                        context_content.append(text)

                else:
                    sp = 0
                    ep = 412
                    scores_arr=[]
                    while (True):
                        if (ep >= text_len and sp < text_len):
                            sub_text = text_id[sp:text_len]
                            concat_text = ques_id + sub_text
                            token_tensor, segment_tensor, mask_tensor = pad_batch_tensorize([concat_text], args.cuda)

                            fw_args = (token_tensor, segment_tensor, mask_tensor)
                            net_out = net(*fw_args)
                            scores_arr.append(net_out[0][0].item())
                            sp += 312
                            ep += 312
                        else:
                            if (ep > text_len):
                                break
                            else:
                                sub_text = text_id[sp:ep]
                                concat_text = ques_id + sub_text
                                token_tensor, segment_tensor, mask_tensor = pad_batch_tensorize([concat_text],
                                                                                                args.cuda)

                                fw_args = (token_tensor, segment_tensor, mask_tensor)
                                net_out = net(*fw_args)
                                scores_arr.append(net_out[0][0].item())
                                sp += 312
                                ep += 312
                    if (max(scores_arr)>highest_score[-1]):
                        highest_score.clear()
                        highest_score.append(net_out[0][0].item())
                        context_new_id.clear()
                        context_new_id.append(new_docid)
                        context_id.clear()
                        context_id.append(docid)
                        context_content.clear()
                        context_content.append(text)
            tmp_dict={}
            tmp_dict['index']=index + 1501
            tmp_dict['id']=id
            tmp_dict['question'] = question_text
            tmp_dict['new_docid']=context_new_id[-1]
            tmp_dict['docid'] = context_id[-1]
            tmp_dict['text'] = context_content[-1]
            with open(join(AFTER_DIR, '{}.json'.format(index + 1501)), 'w',
                      encoding='utf-8') as f:
                json.dump(tmp_dict, f, ensure_ascii=False)


            print('finish processing {}'.format(index+1501))



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