import os
from os.path import exists, join
import random
import torch
import csv
import re
import pickle
import json
from shutil import copyfile
from model.bert_model import BertMatcher
from utils import count_data
import torch
import json
import re
import os
from os.path import join,exists
import argparse
import pyrouge
from mrc.mrc_model import BertReader
from torch.utils.data import DataLoader
from mrc.batcher import coll_fn
from pytorch_pretrained_bert import BertTokenizer
from mrc.batcher import pad_batch_tensorize
from toolz.sandbox import unzip
from rouge import Rouge
from rouge import FilesRouge
import logging
import tempfile
import subprocess as sp

from cytoolz import curry

from pyrouge import Rouge155
from pyrouge.utils import log
from pytorch_pretrained_bert import BertTokenizer
link_dict={}
find_twice=[]
try:
    DATA_DIR = 'comprehension'
    DATASET_DIR = 'data/mrc/'
    TEST_DIR = 'test'
    DIR='data/mrc/train'
    MRC_DIR='data/mrc/'
    CONTEXT_DIR='data/NCPPolicies_context_20200301.csv'
    TRAIN_DIR='data/NCPPolicies_train_20200301.csv'
    REALATE_DIR='data/mrc/relation.pkl'
    TWICE_DIR='data/mrc/twice.pkl'
except KeyError:
    print('please use environment variable to specify data directories')

def filter_text(sentence):
    sub_token = ''
    return re.sub('\s+', sub_token, sentence)

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
def make_over_dir():
    if not exists(MRC_DIR):
        os.makedirs(MRC_DIR)
        print('Dir used for Machine Reading Created ')

def preprocess_context():

    csv_reader = csv.reader(open(CONTEXT_DIR), delimiter='\t')
    rows = [row for row in csv_reader]
    docid_name=rows[0][0]
    text_name=rows[0][1]
    json_context_dirs=join(MRC_DIR,'context')
    tmp_dict = {}
    if not exists(json_context_dirs):
        os.makedirs(json_context_dirs)
    with open(join(MRC_DIR, 'context.txt'), 'w', encoding='utf-8') as fw:
      for i, row_context in enumerate(rows):
          if(i==0):
              continue
          else:
              tmp_dict['new_docid']=i
              tmp_dict[docid_name]=row_context[0]

              data=filter_text(row_context[1].replace(' ','').replace('&ensp;',''))
              tmp_dict[text_name]=data

              with open(join(json_context_dirs,'{}.json'.format(i)),'w',encoding='utf-8') as f:
                  json.dump(tmp_dict,f,ensure_ascii=False)
              json.dump(tmp_dict,fw,ensure_ascii=False)
              fw.write('\n')
              link_dict[row_context[0]]=i
    with open(REALATE_DIR,'wb') as v:
        pickle.dump(link_dict,v)
        print('Relation stored')

    print('Pre-processed context finished')

def process_second_part_example():
    final_path='data/final/test_sample_with_context'
    convert_to_path='data/final/context_for_selection'
    os.makedirs(convert_to_path)
    for index in range(1643):
        with open(join(final_path,'{}.json'.format(index+1))) as f:
            js_data = json.load(f)
            print('loading: {}'.format(index + 1))
            id, question, docid, text=(js_data['id'], js_data['question'], js_data['docid'],js_data['text'])
        tokenizer = BertTokenizer.from_pretrained('./MRC_pretrain')

        text_tok = tokenizer.tokenize(text)
        text_id = tokenizer.convert_tokens_to_ids(text_tok)
        text_len = len(text_id)


        ques_tok = tokenizer.tokenize("[CLS] " + question + " [SEP]")
        ques_id = tokenizer.convert_tokens_to_ids(ques_tok)
        question_len = len(ques_id)

        tmp_dict={}
        tmp_dict['id']=id
        tmp_dict['docid']=docid
        tmp_dict['question'] = ques_id
        tmp_dict['question_length'] = question_len
        tmp_dict['text'] = text_id
        tmp_dict['text_length'] = text_len
        tmp_dict['text_tok'] = text_tok
        tmp_dict['original_text'] = text
        tmp_dict['original_question'] = "[CLS] " + question + " [SEP]"

        with open(join(convert_to_path, '{}.json'.format(index+1)), 'w', encoding='utf-8') as v:
            json.dump(tmp_dict, v, ensure_ascii=False)

def process_mrc_example():
    csv_reader = csv.reader(open(TRAIN_DIR), delimiter='\t')
    rows = [row for row in csv_reader]
    docid_name = rows[0][1]
    question_name = rows[0][2]
    answer_name = rows[0][3]
    json_positive_dirs = join(MRC_DIR, '200_sample')
    if not exists(json_positive_dirs):
        os.makedirs(json_positive_dirs)
        print('Dir used for mrc samples Created ')
    with open(REALATE_DIR,'rb') as v:
        relation_dict=pickle.load(v)
    sample_rows = rows[:200]
    tmp_dict = {}
    count=0
    maxlen = 0
    for i, sample_raw in enumerate(sample_rows):
        if (i == 0):
            continue
        else:
            print('start processing {}'.format(i))

            try:
              new_docid=relation_dict[sample_raw[1]]
              tmp_dict['new_docid'] = new_docid
              with open(join(join(MRC_DIR,'context'),'{}.json'.format(new_docid)),'rb') as p:
                  context=json.load(p)

            except KeyError:
              print('mrc sample {} - related document not found')


            # tmp_dict[docid_name] = sample_raw[1]
            tokenizer = BertTokenizer.from_pretrained('./MRC_pretrain')

            text = context['text']
            text_tok = tokenizer.tokenize(text)
            text_id = tokenizer.convert_tokens_to_ids(text_tok)
            text_len = len(text_id)

            question = filter_text(sample_raw[2].replace(' ', '').replace('&ensp;',''))
            ques_tok = tokenizer.tokenize("[CLS] " + question + " [SEP]")
            ques_id = tokenizer.convert_tokens_to_ids(ques_tok)
            question_len = len(ques_id)
            maxlen = question_len if question_len > maxlen else maxlen

            answer=filter_text(sample_raw[3].replace(' ','').replace('&ensp;',''))
            ans_tok = tokenizer.tokenize(answer)
            ans_id = tokenizer.convert_tokens_to_ids(ans_tok)
            ans_len=len(ans_id)

            suppose_start=[] #可能的start位置
            for i in range(text_len):
                if(text_id[i]==ans_id[0]):
                    suppose_start.append(i)

            s = 0
            e = 0
            if(len(suppose_start)<=0):
                continue

            else:
                for t in range(len(suppose_start)):
                    start=suppose_start[t]
                    end=suppose_start[t]
                    for m in range(ans_len):
                        if(m+start>=text_len):
                            break
                        elif(ans_id[m]==text_id[m+start]):
                            end+=1
                        else:
                            break
                    if(end-start!=ans_len):
                        continue
                    else:
                        s=suppose_start[t]
                        e=end
                        break
            if(s==0 and e==0):
                continue
            else:
                span_arr=[0]*(s-0)+[1]*(e-s)+[0]*(text_len-e)
                assert len(span_arr)==text_len


                tmp_dict['question']=ques_id
                tmp_dict['question_length']=question_len
                tmp_dict['text']=text_id
                tmp_dict['text_length'] = text_len
                tmp_dict['answer_span']=span_arr
                tmp_dict['text_tok'] = text_tok
                tmp_dict['original_text']=text

                with open(join(json_positive_dirs,'{}.json'.format(count)),'w',encoding='utf-8') as f:
                    json.dump(tmp_dict,f,ensure_ascii=False)
                    count+=1





    # print('sample index larger than 512 is {}'.format(count))
    print('Pre-processed {} mrc samples finished' .format(count))



def main(args):
    process_second_part_example()
    convert_to_path = 'data/final/context_for_selection'
    save_path='data/final/context_after_generation'
    os.makedirs(save_path)
    meta = json.load(open(join(DATA_DIR, 'meta.json')))
    nargs = meta['net_args']
    ckpt = load_best_ckpt(DATA_DIR)
    net = BertReader(**nargs)
    net.load_state_dict(ckpt)
    if args.cuda:
        net = net.to('cuda')
    net.eval()

    with torch.no_grad():
        for index in range(1643):

            with open(join(convert_to_path, '{}.json'.format(index+1))) as f:
                js_data = json.load(f)
                print('loading: {}'.format(index+1 ))
                id, docid, question, question_length, text, text_length, text_tok,original_text,original_question = (js_data['id'], js_data['docid'],
                js_data['question'], js_data['question_length'], js_data['text'], js_data['text_length'],
                js_data['text_tok'],js_data['original_text'],js_data['original_question'])




            if (question_length + text_length <= 512):
                concat_text = question + text
                token_tensor, segment_tensor, mask_tensor = pad_batch_tensorize([concat_text], args.cuda)
                question_lengths = torch.tensor([question_length])
                question_lengths = question_lengths.cuda()

                text_lengths = torch.tensor([text_length])
                text_lengths = text_lengths.cuda()

                fw_args = (token_tensor, segment_tensor, mask_tensor, question_lengths, text_lengths)
                net_out = net(*fw_args)

                net_out = torch.squeeze(net_out)
                net_out = net_out[question_length:question_length + text_length]
                leng = net_out.size(0)
                propuse = []
                for i in range(leng):
                    if (net_out[i].item() > 0.5):
                        propuse.append(1)
                    else:
                        propuse.append(0)

                if(not(1 in propuse)):
                    propuse.clear()
                    for i in range(leng):
                        if (net_out[i].item() > 1e-4):
                            propuse.append(1)
                        else:
                            propuse.append(0)
                bulid = []
                output=''


                for t in range(len(propuse)):
                    if (propuse[t] == 1):

                        bulid.append(text[t])

                        output+=text_tok[t] if(text_tok[t]!='[UNK]') else ''
                output = output.replace('##', '')
                print(output)


                tmp_dict={}
                tmp_dict['id'] = id
                tmp_dict['docid'] = docid
                tmp_dict['answer']=str(output)
                with open(join(save_path, '{}.json'.format(index + 1)), 'w', encoding='utf-8') as v:
                    json.dump(tmp_dict, v, ensure_ascii=False)

            else:
                sp = 0
                ep = 412
                sub_text_arr = []
                sub_text_length_arr = []
                start_index = []
                while (True):
                    if (ep >= text_length and sp < text_length):

                        sub_text = text[sp:text_length]
                        sub_text_arr.append(sub_text)

                        sub_text_length = text_length - sp
                        sub_text_length_arr.append(sub_text_length)
                        start_index.append(sp)

                        assert question_length + text_length - sp <= 512
                        sp += 312
                        ep += 312

                    else:
                        if (ep > text_length):
                            break
                        else:
                            sub_text = text[sp:ep]
                            sub_text_arr.append(sub_text)

                            sub_text_length = ep - sp
                            sub_text_length_arr.append(sub_text_length)
                            start_index.append(sp)

                            assert question_length + ep - sp <= 512

                            sp += 312
                            ep += 312

                meta_s = json.load(open(join('matcher', 'meta.json')))
                nargs_s = meta_s['net_args']
                ckpt_s = load_best_ckpt('matcher')
                net_s = BertMatcher(**nargs_s)
                net_s.load_state_dict(ckpt_s)
                if args.cuda:
                    net_s = net_s.cuda()
                net_s.eval()
                with torch.no_grad():
                    highest_score = [0]
                    current=-1
                    for i in range(len(sub_text_arr)):

                        concat_text = question + sub_text_arr[i]
                        token_tensor, segment_tensor, mask_tensor = pad_batch_tensorize([concat_text], args.cuda)
                        fw_args = (token_tensor, segment_tensor, mask_tensor)
                        net_out = net_s(*fw_args)
                        if (net_out[0][0].item() > highest_score[-1]):
                            highest_score.clear()
                            highest_score.append(net_out[0][0].item())
                            current=i
                used_text=sub_text_arr[current]

                propuse = [0] * text_length



                concat_text = question +used_text

                token_tensor, segment_tensor, mask_tensor = pad_batch_tensorize([concat_text], args.cuda)
                question_lengths = torch.tensor([question_length])
                question_lengths = question_lengths.cuda()

                text_lengths = torch.tensor([sub_text_length_arr[current]])
                text_lengths = text_lengths.cuda()

                fw_args = (token_tensor, segment_tensor, mask_tensor, question_lengths, text_lengths)
                net_out = net(*fw_args)

                net_out = torch.squeeze(net_out)
                net_out = net_out[question_length:question_length + text_length]
                leng = net_out.size(0)

                for ga in range(leng):

                    if (net_out[ga].item() > 0.5):

                        propuse[ga+start_index[current]] = 1

                if(not(1 in propuse)):
                    for ga in range(leng):

                        if (net_out[ga].item() > 1e-4):
                            propuse[ga + start_index[current]] = 1

                bulid = []
                output = ''



                for t in range(len(propuse)):
                    if (propuse[t] == 1):
                        bulid.append(text[t])
                        output += text_tok[t] if (text_tok[t] != '[UNK]') else ''
                output=output.replace('##','')
                print(output)


                tmp_dict = {}
                tmp_dict['id'] = id
                tmp_dict['docid'] = docid
                tmp_dict['answer'] = str(output)
                with open(join(save_path, '{}.json'.format(index + 1)), 'w', encoding='utf-8') as v:
                    json.dump(tmp_dict, v, ensure_ascii=False)



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
