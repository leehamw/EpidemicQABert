import pandas as pd
import numpy as np
import os
from os.path import exists, join
import multiprocessing as mp
import random
import torch
import csv
import re
import pickle
import json
from pytorch_pretrained_bert import BertTokenizer
from shutil import copyfile
from utils import count_data
from gensim.summarization import bm25
from gensim import corpora
import jieba
import heapq
STOP_DIR = 'data/stopword.txt'
link_dict={}



try:
    CLASSIFICATION_DIR='data/class/'
    CONTEXT_DIR='data/NCPPolicies_context_20200301.csv'
    TRAIN_DIR='data/NCPPolicies_train_20200301.csv'
    REALATE_DIR='data/class/relation.pkl'
except KeyError:
    print('please use environment variable to specify data directories')
def stopwordlist():
    stopwords=[line.strip() for line in open(STOP_DIR, 'r', encoding='utf-8').readlines()]
    return stopwords
def filter_text(sentence):
    sub_token = ''
    return re.sub('\s+', sub_token, sentence)

def make_over_dir():
    if not exists(CLASSIFICATION_DIR):
        os.makedirs(CLASSIFICATION_DIR)
        print('Dir used for Classification Created ')

def preprocess_context():

    csv_reader = csv.reader(open(CONTEXT_DIR), delimiter='\t')
    rows = [row for row in csv_reader]
    docid_name=rows[0][0]
    text_name=rows[0][1]
    json_context_dirs=join(CLASSIFICATION_DIR,'context')
    tmp_dict = {}
    if not exists(json_context_dirs):
        os.makedirs(json_context_dirs)
    with open(join(CLASSIFICATION_DIR, 'context.txt'), 'w', encoding='utf-8') as fw:
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

def process_positive_example_v1():
    stopwords = stopwordlist()
    os.makedirs('data/class/pos')
    os.makedirs('data/class/neg')
    context_path = 'data/class/context'
    context_data = count_data(context_path)
    corpus = []
    new_docid_arr = []
    for i in range(context_data):
        with open(join('data/class/context', '{}.json'.format(i + 1))) as f:
            js_data = json.load(f)
            text = filter_text(
                js_data['text'].replace(' ', '').replace('&ensp;', '').replace('&rbsp;', '').replace('&mbsp;', ''))
            new_docid = js_data['new_docid']
        data = list(jieba.lcut(filter_text(text), cut_all=False, HMM=True))
        remove = lambda token: False if token in stopwords else True
        data = list(filter(remove, data))
        print(new_docid)
        corpus.append(data)
        new_docid_arr.append(new_docid)
    dictionary = corpora.Dictionary(corpus)
    bm25Model = bm25.BM25(corpus)

    csv_reader = csv.reader(open(TRAIN_DIR), delimiter='\t')
    rows = [row for row in csv_reader]
    docid_name = rows[0][1]
    question_name = rows[0][2]
    answer_name = rows[0][3]
    json_positive_dirs = join(CLASSIFICATION_DIR, 'positive_sample')
    # if not exists(json_positive_dirs):
    #     os.makedirs(json_positive_dirs)
    #     print('Dir used for positive samples Created ')
    with open(REALATE_DIR,'rb') as v:
        relation_dict=pickle.load(v)
    sample_rows = rows[:1582] + rows[1583:1955] + rows[1956:3781] + rows[3782:]

    maxlen=0
    count=1
    ncount=1
    right=0
    for i, sample_raw in enumerate(sample_rows):
        print('loading {}'.format(i))
        if (i == 0):
            continue
        else:
            tokenizer = BertTokenizer.from_pretrained('./MRC_pretrain')
            new_docid=relation_dict[sample_raw[1]]
            remove = lambda token: False if token in stopwords else True
            question = filter_text(sample_raw[2].replace(' ', '').replace('&ensp;', ''))
            q_data = list(jieba.lcut(filter_text(question), cut_all=False, HMM=True))
            q_data = list(filter(remove, q_data))
            scores = bm25Model.get_scores(q_data)
            max_num_index_list = map(scores.index, heapq.nlargest(3, scores))
            max_num_index_list = list(max_num_index_list)
            arr = []
            for m in max_num_index_list:
                idx = m
                fname = new_docid_arr[idx]
                arr.append(fname)
            if (not(new_docid in arr)):
                continue
            else:
                right+=1

                for con in arr:
                    with open(join('data/class/context', '{}.json'.format(con))) as c:
                        cn_data = json.load(c)
                        cont_docid, docid, text = (cn_data['new_docid'], cn_data['docid'], cn_data['text'])

                    ques_tok = tokenizer.tokenize("[CLS] " + question + " [SEP]")
                    ques_id = tokenizer.convert_tokens_to_ids(ques_tok)
                    question_len = len(ques_id)

                    text_tok = tokenizer.tokenize(text)
                    text_id = tokenizer.convert_tokens_to_ids(text_tok)
                    text_len = len(text_id)
                    if (con == new_docid):
                        tmp_dict = {}
                        tmp_dict['is_related'] = 1
                        tmp_dict['new_docid'] = new_docid

                        answer = filter_text(sample_raw[3].replace(' ', '').replace('&ensp;', ''))

                        ans_tok = tokenizer.tokenize(answer)
                        ans_id = tokenizer.convert_tokens_to_ids(ans_tok)
                        ans_len = len(ans_id)
                        suppose_start = []  # 可能的start位置
                        for i in range(text_len):
                            if (text_id[i] == ans_id[0]):
                                suppose_start.append(i)

                        s = 0
                        e = 0
                        if (len(suppose_start) <= 0):
                            continue

                        else:
                            for t in range(len(suppose_start)):
                                start = suppose_start[t]
                                end = suppose_start[t]
                                for m in range(ans_len):
                                    if (m + start >= text_len):
                                        break
                                    elif (ans_id[m] == text_id[m + start]):
                                        end += 1
                                    else:
                                        break
                                if (end - start != ans_len):
                                    continue
                                else:
                                    s = suppose_start[t]
                                    e = end
                                    break
                        if (s == 0 and e == 0):
                            continue
                        else:
                            span_arr = [0] * (s - 0) + [1] * (e - s) + [0] * (text_len - e)

                        if (question_len + text_len <= 512):

                            tmp_dict['question'] = ques_id

                            tmp_dict['text'] = text_id

                            with open(join('data/class/pos', '{}.json'.format(count)), 'w', encoding='utf-8') as f:
                                json.dump(tmp_dict, f, ensure_ascii=False)
                                count += 1
                        else:
                            sp = 0
                            ep = 412
                            assert question_len <= 100 and text_len >= 412
                            while (True):
                                if (ep >= text_len and sp < text_len):

                                    sub_text = text_id[sp:text_len]
                                    tmp_dict['question'] = ques_id

                                    tmp_dict['text'] = sub_text

                                    assert question_len + text_len - sp <= 512

                                    with open(join('data/class/pos', '{}.json'.format(count)), 'w',
                                              encoding='utf-8') as f:
                                        json.dump(tmp_dict, f, ensure_ascii=False)
                                        count += 1

                                    sp += 312
                                    ep += 312
                                # else:
                                #         break
                                else:
                                    if (ep > text_len):
                                        break
                                    else:
                                        sub_text = text_id[sp:ep]
                                        tmp_dict['question'] = ques_id

                                        tmp_dict['text'] = sub_text

                                        assert question_len + ep - sp <= 512

                                        with open(join('data/class/pos', '{}.json'.format(count)), 'w',
                                                  encoding='utf-8') as f:
                                            json.dump(tmp_dict, f, ensure_ascii=False)
                                            count += 1

                                        sp += 312
                                        ep += 312
                    else:
                        tmp_dict['is_related'] = 0
                        tmp_dict['new_docid'] = con
                        tmp_dict['question'] = ques_id
                        if (question_len + text_len <= 512):

                            tmp_dict['text'] = text_id

                            with open(join('data/class/neg', '{}.json'.format(ncount)), 'w', encoding='utf-8') as f:
                                json.dump(tmp_dict, f, ensure_ascii=False)
                                ncount += 1

                        else:

                            sp = 0
                            ep = 412
                            assert question_len <= 100 and text_len >= 412
                            while (True):
                                if (ep >= text_len and sp < text_len):

                                    sub_text = text_id[sp:text_len]
                                    tmp_dict['question'] = ques_id

                                    tmp_dict['text'] = sub_text

                                    assert question_len + text_len - sp <= 512

                                    with open(join('data/class/neg', '{}.json'.format(ncount)), 'w',
                                              encoding='utf-8') as f:
                                        json.dump(tmp_dict, f, ensure_ascii=False)
                                        ncount += 1

                                    sp += 312
                                    ep += 312
                                # else:
                                #         break
                                else:
                                    if (ep > text_len):
                                        break
                                    else:
                                        sub_text = text_id[sp:ep]
                                        tmp_dict['question'] = ques_id

                                        tmp_dict['text'] = sub_text

                                        assert question_len + ep - sp <= 512

                                        with open(join('data/class/neg', '{}.json'.format(ncount)), 'w',
                                                  encoding='utf-8') as f:
                                            json.dump(tmp_dict, f, ensure_ascii=False)
                                            ncount += 1

                                        sp += 312
                                        ep += 312











    print('Pre-processed {} positive samples finished'.format(right))
    print(len(sample_rows))


def process_positive_example():

    csv_reader = csv.reader(open(TRAIN_DIR), delimiter='\t')
    rows = [row for row in csv_reader]
    docid_name = rows[0][1]
    question_name = rows[0][2]
    answer_name = rows[0][3]
    json_positive_dirs = join(CLASSIFICATION_DIR, 'positive_sample')
    if not exists(json_positive_dirs):
        os.makedirs(json_positive_dirs)
        print('Dir used for positive samples Created ')
    with open(REALATE_DIR, 'rb') as v:
        relation_dict = pickle.load(v)
    sample_rows = rows[:1582] + rows[1583:1955] + rows[1956:3781] + rows[3782:]
    tmp_dict = {}
    maxlen = 0
    count = 1
    for i, sample_raw in enumerate(sample_rows):
        if (i == 0):
            continue
        else:
            tmp_dict['is_related'] = 1
            try:
                new_docid = relation_dict[sample_raw[1]]
                tmp_dict['new_docid'] = new_docid
                with open(join(join(CLASSIFICATION_DIR, 'context'), '{}.json'.format(new_docid)), 'rb') as p:
                    context = json.load(p)
                # original_text=context['text']

                # tmp_dict['related_context']=
            except KeyError:
                print('positive sample {} - related document not found')
                # new_docid=0
                # tmp_dict['new_docid'] = new_docid
                # tmp_dict['related_context'] =''

            tmp_dict[docid_name] = sample_raw[1]

            # question = filter_text(sample_raw[2].replace(' ', '').replace('&ensp;',''))

            # tmp_dict[question_name] = question
            # answer=filter_text(sample_raw[3].replace(' ','').replace('&ensp;',''))

            # tmp_dict[answer_name] = answer

            tokenizer = BertTokenizer.from_pretrained('./MRC_pretrain')

            text = context['text']
            text_tok = tokenizer.tokenize(text)
            text_id = tokenizer.convert_tokens_to_ids(text_tok)
            text_len = len(text_id)

            question = filter_text(sample_raw[2].replace(' ', '').replace('&ensp;', ''))
            ques_tok = tokenizer.tokenize("[CLS] " + question + " [SEP]")
            ques_id = tokenizer.convert_tokens_to_ids(ques_tok)
            question_len = len(ques_id)
            maxlen = question_len if question_len > maxlen else maxlen

            answer = filter_text(sample_raw[3].replace(' ', '').replace('&ensp;', ''))

            ans_tok = tokenizer.tokenize(answer)
            ans_id = tokenizer.convert_tokens_to_ids(ans_tok)
            ans_len = len(ans_id)
            suppose_start = []  # 可能的start位置
            for i in range(text_len):
                if (text_id[i] == ans_id[0]):
                    suppose_start.append(i)

            s = 0
            e = 0
            if (len(suppose_start) <= 0):
                continue

            else:
                for t in range(len(suppose_start)):
                    start = suppose_start[t]
                    end = suppose_start[t]
                    for m in range(ans_len):
                        if (m + start >= text_len):
                            break
                        elif (ans_id[m] == text_id[m + start]):
                            end += 1
                        else:
                            break
                    if (end - start != ans_len):
                        continue
                    else:
                        s = suppose_start[t]
                        e = end
                        break
            if (s == 0 and e == 0):
                continue
            else:
                span_arr = [0] * (s - 0) + [1] * (e - s) + [0] * (text_len - e)

            if (question_len + text_len <= 512):

                tmp_dict['question'] = ques_id

                tmp_dict['text'] = text_id

                with open(join(json_positive_dirs, '{}.json'.format(count)), 'w', encoding='utf-8') as f:
                    json.dump(tmp_dict, f, ensure_ascii=False)
                    count += 1
            else:
                sp = 0
                ep = 412
                assert question_len <= 100 and text_len >= 412
                while (True):
                    if (ep >= text_len and sp < text_len):

                        sub_text = text_id[sp:text_len]
                        tmp_dict['question'] = ques_id

                        tmp_dict['text'] = sub_text

                        assert question_len + text_len - sp <= 512

                        with open(join(json_positive_dirs, '{}.json'.format(count)), 'w',
                                  encoding='utf-8') as f:
                            json.dump(tmp_dict, f, ensure_ascii=False)
                            count += 1

                        sp += 312
                        ep += 312
                    # else:
                    #         break
                    else:
                        if (ep > text_len):
                            break
                        else:
                            sub_text = text_id[sp:ep]
                            tmp_dict['question'] = ques_id

                            tmp_dict['text'] = sub_text

                            assert question_len + ep - sp <= 512

                            with open(join(json_positive_dirs, '{}.json'.format(count)), 'w',
                                      encoding='utf-8') as f:
                                json.dump(tmp_dict, f, ensure_ascii=False)
                                count += 1

                            sp += 312
                            ep += 312
                            # else:
                            #     sp += 312
                            #     ep += 312

    print('Pre-processed {} positive samples finished'.format(count))
    print(maxlen)


def process_negative_example():
    num_positive_data=count_data(join(CLASSIFICATION_DIR,'positive_sample'))
    negative_path=join(CLASSIFICATION_DIR,'negative_sample')
    if not exists(negative_path):
        os.makedirs(negative_path)
    count=1
    for num_negative in range(num_positive_data):
        for m in range(9):


            random.seed()
            tmp_dict = {}
            with open(join(join(CLASSIFICATION_DIR,'positive_sample'),'{}.json'.format(num_negative+1))) as f:
                positive_sample=json.load(f)
            tmp_dict['is_related'] = 0
            proper_docid=positive_sample['new_docid']


            question=positive_sample['question']
        # answer=positive_sample['answer']

            wrong_docid=random.randint(1,num_positive_data)
            while(wrong_docid==proper_docid):
                wrong_docid = random.randint(1, num_positive_data)
            with open(join(join(CLASSIFICATION_DIR, 'context'), '{}.json'.format(wrong_docid))) as n:
                negative_sample=json.load(n)

            tokenizer = BertTokenizer.from_pretrained('./MRC_pretrain')
            wrong_context=negative_sample['text']
            text_tok = tokenizer.tokenize(wrong_context)
            text_id = tokenizer.convert_tokens_to_ids(text_tok)
            text_len = len(text_id)

            tmp_dict['new_docid'] = wrong_docid
            tmp_dict['docid'] = negative_sample['docid']



            tmp_dict['question'] = question
            question_len = len(question)
            if (question_len + text_len <= 512):


                tmp_dict['text'] = text_id

                with open(join(negative_path, '{}.json'.format(count)), 'w', encoding='utf-8') as f:
                    json.dump(tmp_dict, f, ensure_ascii=False)
                count+=1

            else:
                inde=text_len-412
                id = random.randint(0, inde)
                tmp_dict['text'] = text_id[id:id+412]
                with open(join(negative_path, '{}.json'.format(count)), 'w', encoding='utf-8') as f:
                    json.dump(tmp_dict, f, ensure_ascii=False)
                count += 1

    print('Pre-processed negative samples finished')

def random_shuffle_data():
    num_positive_data = count_data(join(CLASSIFICATION_DIR, 'pos'))
    num_negative_data = count_data(join(CLASSIFICATION_DIR, 'neg'))
    positive_path=join(CLASSIFICATION_DIR, 'pos')
    negative_path=join(CLASSIFICATION_DIR,'neg')
    overall_path=join(CLASSIFICATION_DIR,'sample')
    if not exists(overall_path):
        os.makedirs(overall_path)

    positive_arr=[i+1 for i in range(num_positive_data)]
    negative_arr=[n+1 for n in range(num_negative_data)]
    random.seed()
    random.shuffle(positive_arr)

    random.seed()
    random.shuffle(negative_arr)

    overall_arr=[1]*len(positive_arr)+[0]*len(negative_arr)
    random.seed()
    random.shuffle(overall_arr)

    positive_index=0
    negative_index=0
    for o in range(len(overall_arr)):
        pos_or_neg=overall_arr[o]
        if(pos_or_neg==1):
            copyfile(join(positive_path,'{}.json'.format(positive_arr[positive_index])),join(overall_path,'{}.json'.format(o+1)))
            positive_index+=1
            assert positive_index <=num_positive_data
        else:
            copyfile(join(negative_path, '{}.json'.format(negative_arr[negative_index])),join(overall_path, '{}.json'.format(o+1)))
            negative_index+=1
            assert negative_index <=num_negative_data
    print('Finished shuffling data')

def split_data():
    sample_path = join(CLASSIFICATION_DIR, 'sample')
    split=['train','valid','test']
    # print(count_data((join(CLASSIFICATION_DIR, 'negative_sample'))))
    print(count_data(sample_path))
    num_sample_data=[i+1for i in range(count_data(sample_path))]
    data_split=[num_sample_data[:52000],num_sample_data[52000:54000],num_sample_data[54000:]]
    for path in split:
        if not exists(join(CLASSIFICATION_DIR,path)):
            os.makedirs(join(CLASSIFICATION_DIR,path))
    for i,w in enumerate(data_split):
        for index,data in enumerate(w):
            copyfile(join(sample_path, '{}.json'.format(data)),
                     join(join(CLASSIFICATION_DIR,split[i]), '{}.json'.format(index + 1)))
    print('Finished spliting data')




















def main():
    # make_over_dir()
    # preprocess_context()
    process_positive_example_v1()

    # random_shuffle_data()
    # split_data()




if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda._initialized = True
    print(torch.cuda.is_available())

    main()


