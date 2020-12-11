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
    FINAL_DIR='data/final'

    CONTEXT_DIR='data/NCPPolicies_context_20200301.csv'
    TEST_DIR='data/NCPPolicies_test.csv'
    REALATE_DIR='data/final/relation.pkl'

except KeyError:
    print('please use environment variable to specify data directories')

def filter_text(sentence):
    sub_token = ''
    return re.sub('\s+', sub_token, sentence)

def make_over_dir():
    if not exists(FINAL_DIR):
        os.makedirs(FINAL_DIR)
        print('Dir used for Final test Created ')

def preprocess_context():

    csv_reader = csv.reader(open(CONTEXT_DIR), delimiter='\t')
    rows = [row for row in csv_reader]
    docid_name=rows[0][0]
    text_name=rows[0][1]
    json_context_dirs=join(FINAL_DIR,'context')
    tmp_dict = {}
    if not exists(json_context_dirs):
        os.makedirs(json_context_dirs)
    with open(join(FINAL_DIR, 'context.txt'), 'w', encoding='utf-8') as fw:
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



def process_test_example():
    csv_reader = csv.reader(open(TEST_DIR), delimiter='\t')
    rows = [row for row in csv_reader]



    json_positive_dirs = join(FINAL_DIR, 'original_test_sample')
    if not exists(json_positive_dirs):
        os.makedirs(json_positive_dirs)
        print('Dir used for mrc samples Created ')
    # with open(REALATE_DIR,'rb') as v:
    #     relation_dict=pickle.load(v)

    tmp_dict = {}
    maxlen=0
    for i, sample_raw in enumerate(rows):
        if (i == 0):
            continue
        else:
            print('start processing {}'.format(i))

            tmp_dict['index']=i
            tmp_dict['id']=filter_text(sample_raw[0].replace(' ', '').replace('&ensp;',''))
            original_question_text=filter_text(sample_raw[1].replace(' ', '').replace('&ensp;',''))
            tmp_dict['question']=original_question_text

            tokenizer = BertTokenizer.from_pretrained('./MRC_pretrain')


            question = filter_text(sample_raw[1].replace(' ', '').replace('&ensp;',''))
            ques_tok = tokenizer.tokenize("[CLS] " + question + " [SEP]")
            tmp_dict['question_token'] = ques_tok
            ques_id = tokenizer.convert_tokens_to_ids(ques_tok)
            tmp_dict['question_id']=ques_id
            question_len = len(ques_id)
            maxlen=question_len  if question_len > maxlen else maxlen
            tmp_dict['token_question_length'] = question_len




            with open(join(json_positive_dirs, '{}.json'.format(i)), 'w',
                                              encoding='utf-8') as f:
                json.dump(tmp_dict, f, ensure_ascii=False)









    print('Pre-processed test samples finished' )
    print(maxlen)



def main():
    make_over_dir()
    # preprocess_context()
    process_test_example()




if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda._initialized = True
    print(torch.cuda.is_available())

    main()

