import os
from os.path import exists, join
import random
import torch
import csv
import re
import pickle
import json
from shutil import copyfile
from utils import count_data
from pytorch_pretrained_bert import BertTokenizer
link_dict={}
find_twice=[]
try:
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



def process_mrc_example():
    csv_reader = csv.reader(open(TRAIN_DIR), delimiter='\t')
    rows = [row for row in csv_reader]
    docid_name = rows[0][1]
    question_name = rows[0][2]
    answer_name = rows[0][3]
    json_positive_dirs = join(MRC_DIR, 'mrc_sample')
    if not exists(json_positive_dirs):
        os.makedirs(json_positive_dirs)
        print('Dir used for mrc samples Created ')
    with open(REALATE_DIR,'rb') as v:
        relation_dict=pickle.load(v)
    sample_rows = rows[:1582] + rows[1583:1955] + rows[1956:3781] + rows[3782:]
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
                if(question_len+text_len<=512):

                    tmp_dict['question']=ques_id
                    tmp_dict['question_length']=question_len
                    tmp_dict['text']=text_id
                    tmp_dict['text_length'] = text_len
                    tmp_dict['answer_span']=span_arr
                    tmp_dict['text_tok'] = text_tok
                    tmp_dict['original_text']=text
                    tmp_dict['start_index'] = s
                    tmp_dict['end_index'] = e


                    with open(join(json_positive_dirs,'{}.json'.format(count)),'w',encoding='utf-8') as f:
                        json.dump(tmp_dict,f,ensure_ascii=False)
                        count+=1
                else:
                    sp=0
                    ep=412
                    assert question_len<=100 and text_len>=412
                    while(True):
                        if(ep>=text_len and sp<text_len):

                            sub_text=text_id[sp:text_len]
                            tmp_dict['question'] = ques_id
                            tmp_dict['question_length'] = question_len
                            tmp_dict['text'] = sub_text
                            tmp_dict['text_length'] = text_len-sp
                            tmp_dict['answer_span'] = span_arr[sp:text_len]
                            tmp_dict['text_tok'] = text_tok[sp:text_len]
                            tmp_dict['original_text'] = text
                            assert text_len-sp==len(span_arr[sp:text_len])

                            assert  question_len+ text_len-sp<=512

                            if(1 in span_arr[sp:text_len] ):

                                with open(join(json_positive_dirs, '{}.json'.format(count)), 'w',
                                          encoding='utf-8') as f:
                                    json.dump(tmp_dict, f, ensure_ascii=False)
                                    count += 1
                                    sp += 312
                                    ep += 312
                            else:
                                break
                        else:
                            if(ep>text_len):
                                break
                            else:
                                sub_text = text_id[sp:ep]
                                tmp_dict['question'] = ques_id
                                tmp_dict['question_length'] = question_len
                                tmp_dict['text'] = sub_text
                                tmp_dict['text_length'] = ep - sp
                                tmp_dict['answer_span'] = span_arr[sp:ep]
                                tmp_dict['text_tok'] = text_tok[sp:ep]
                                tmp_dict['original_text'] = text
                                assert ep - sp==len(span_arr[sp:ep])
                                assert question_len + ep - sp <= 512
                                if(1 in span_arr[sp:ep]):

                                    with open(join(json_positive_dirs, '{}.json'.format(count)), 'w',
                                              encoding='utf-8') as f:
                                        json.dump(tmp_dict, f, ensure_ascii=False)
                                        count += 1
                                    sp+=312
                                    ep+=312
                                else:
                                    sp += 312
                                    ep += 312







    # print('sample index larger than 512 is {}'.format(count))
    print('Pre-processed {} mrc samples finished' .format(count))
    print(maxlen)

def split_data():
    sample_path = join(MRC_DIR, 'mrc_sample')
    split=['train','valid','test']
    num_sample_data=[i  for i in range(count_data(sample_path))]
    data_split=[num_sample_data[:6432],num_sample_data[6432:6632],num_sample_data[6632:]]
    for path in split:
        if not exists(join(MRC_DIR,path)):
            os.makedirs(join(MRC_DIR,path))
    for i,w in enumerate(data_split):
        for index,data in enumerate(w):
            copyfile(join(sample_path, '{}.json'.format(data)),
                     join(join(MRC_DIR,split[i]), '{}.json'.format(index)))
    print('Finished spliting data ')


def main():
    make_over_dir()
    preprocess_context()
    process_mrc_example()
    split_data()
    # for i in range(4300):
    #     with open(join(DIR,'{}.json'.format(i))) as f:
    #         sample=json.load(f)
    #     related_context, question, index , answer= (sample['related_context'], sample['question'], sample['index'], sample['answer'])
    #     tokenizer = BertTokenizer.from_pretrained('./MRC_pretrain')
    #
    #     ques_tok = tokenizer.tokenize("[CLS] " + question + " [SEP]")
    #     ques_id = tokenizer.convert_tokens_to_ids(ques_tok)
    #     question_len = len(ques_id)
    #
    #     text_tok = tokenizer.tokenize(related_context)
    #     text_id = tokenizer.convert_tokens_to_ids(text_tok)
    #     text_len = len(text_id)
    #
    #     ans_tok=tokenizer.tokenize(answer)
    #     ans_id=tokenizer.convert_tokens_to_ids(ans_tok)
    #
    #
    #     print(ans_tok)
    #     print(ans_id)
    #
    #     print(len(question))
    #     print(question)
    #
    #     print(ques_tok)
    #     print(ques_id)
    #     print(question_len)


        # if (question_len != len(question) + 2):
        #     print(len(question))
        #     print(question)
        #
        #     print(ques_tok)
        #     print(ques_id)
        #     print(question_len)
        #     print(i)
        # if (text_len != len(related_context)):
        #     print(len(related_context))
        #     print(related_context)
        #     print(text_tok)
        #     print(text_id)
        #     print(text_len)
        #     print(i)
        # break



if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda._initialized = True
    print(torch.cuda.is_available())

    main()

