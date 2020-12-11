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
try:
    DATA_DIR = 'comprehension'
    DATASET_DIR = 'data/mrc/'
    TEST_DIR = 'test'

    _ROUGE_PATH = '/home/wanglihan/ROUGE/RELEASE-1.5.5/'
    _METEOR_PATH = 'meteor-1.5/meteor-1.5.jar'
except KeyError:
    print('please use environment variable to specify data directories')
def eval_meteor(dec_pattern, dec_dir, ref_pattern, ref_dir):
    """ METEOR evaluation"""
    assert _METEOR_PATH is not None
    ref_matcher = re.compile(ref_pattern)
    refs = sorted([r for r in os.listdir(ref_dir) if ref_matcher.match(r)],
                  key=lambda name: int(name.split('.')[0]))
    dec_matcher = re.compile(dec_pattern)
    decs = sorted([d for d in os.listdir(dec_dir) if dec_matcher.match(d)],
                  key=lambda name: int(name.split('.')[0]))
    @curry
    def read_file(file_dir, file_name):
        with open(join(file_dir, file_name)) as f:
            return ' '.join(f.read().split())
    with tempfile.TemporaryDirectory() as tmp_dir:
        with open(join(tmp_dir, 'ref.txt'), 'w') as ref_f,\
             open(join(tmp_dir, 'dec.txt'), 'w') as dec_f:
            ref_f.write('\n'.join(map(read_file(ref_dir), refs)) + '\n')
            dec_f.write('\n'.join(map(read_file(dec_dir), decs)) + '\n')

        cmd = 'java -Xmx2G -jar {} {} {} -l en -norm'.format(
            _METEOR_PATH, join(tmp_dir, 'dec.txt'), join(tmp_dir, 'ref.txt'))
        output = sp.check_output(cmd.split(' '), universal_newlines=True)
    return output
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
def eval_rouge(dec_pattern, dec_dir, ref_pattern, ref_dir,
               cmd='-c 95 -r 1000 -n 2 -m', system_id=1):
    """ evaluate by original Perl implementation"""
    # silence pyrouge logging
    assert _ROUGE_PATH is not None
    log.get_global_console_logger().setLevel(logging.WARNING)
    with tempfile.TemporaryDirectory() as tmp_dir:
        Rouge155.convert_summaries_to_rouge_format(
            dec_dir, join(tmp_dir, 'dec'))
        Rouge155.convert_summaries_to_rouge_format(
            ref_dir, join(tmp_dir, 'ref'))
        Rouge155.write_config_static(
            join(tmp_dir, 'dec'), dec_pattern,
            join(tmp_dir, 'ref'), ref_pattern,
            join(tmp_dir, 'settings.xml'), system_id
        )
        cmd = (join(_ROUGE_PATH, 'ROUGE-1.5.5.pl')
               + ' -e {} '.format(join(_ROUGE_PATH, 'data'))
               + cmd
               + ' -a {}'.format(join(tmp_dir, 'settings.xml')))
        output = sp.check_output(cmd.split(' '), universal_newlines=True)
    return output

def main(args):
    if not exists(TEST_DIR):
        os.makedirs(TEST_DIR)
    if not exists(os.path.join(TEST_DIR, "reference")):
        os.makedirs(os.path.join(TEST_DIR, "reference"))
    if not exists(os.path.join(TEST_DIR, "decoded")):
        os.makedirs(os.path.join(TEST_DIR, "decoded"))

    meta = json.load(open(join(DATA_DIR, 'meta.json')))
    nargs = meta['net_args']
    ckpt = load_best_ckpt(DATA_DIR)
    net = BertReader(**nargs)
    net.load_state_dict(ckpt)
    if args.cuda:
        net = net.to('cuda')
    net.eval()
    tokenizer = BertTokenizer.from_pretrained('./MRC_pretrain')
    count = 0
    bulids=[]
    answers=[]
    with torch.no_grad():
        for index in range(200):

            with open(join(join(DATASET_DIR, 'test'), '{}.json'.format(index))) as f:
                js_data = json.load(f)
                print('loading: {}'.format(index ))
                question, question_length, text, text_length, answer_span = (
                js_data['question'], js_data['question_length'], js_data['text'], js_data['text_length'],
                js_data['answer_span'])

            concat_text = question + text

            token_tensor, segment_tensor, mask_tensor = pad_batch_tensorize([concat_text], args.cuda)
            question_lengths = torch.tensor([question_length])
            question_lengths = question_lengths.cuda()

            text_lengths = torch.tensor([text_length])
            text_lengths = text_lengths.cuda()

            fw_args = (token_tensor, segment_tensor, mask_tensor, question_lengths, text_lengths)
            net_out = net(*fw_args)

            net_out=torch.squeeze(net_out)
            net_out=net_out[question_length:question_length+text_length]

            leng=net_out.size(0)
            print('leng{}'.format(leng))
            propuse=[]
            for i in range(leng):
                if(net_out[i].item()>0.6):
                    propuse.append(1)
                else:
                    propuse.append(0)
            bulid=[]

            print('text_length{}'.format(text_length))

            for t in range(len(propuse)):
                if(propuse[t]==1):

                    bulid.append(text[t])

            bulid = [str(x) for x in bulid]
            bulids.append(bulid)

            answer_index=answer_span.index(1)
            one=0
            for o in range(len(answer_span)):
                if answer_span[o]==1:
                    one+=1
            print(one)
            answer=text[answer_index:answer_index+one]
            answer=[str(x) for x in answer]
            answers.append(answer)

            with open(join(os.path.join(TEST_DIR, "decoded"),"%d_decoded.txt" % index), 'w') as f:
                # for i, item in enumerate(bulids):

                f.write(' '.join(bulid))

            with open(join(os.path.join(TEST_DIR, "reference"),"%d_reference.txt" % index), 'w', ) as f:
          # for i, item in enumerate(answers):
          #   print(item)
                f.write(' '.join(answer))



    r = pyrouge.Rouge155('/home/wanglihan/ROUGE/RELEASE-1.5.5/')
    r.model_filename_pattern = '#ID#_reference.txt'
    r.system_filename_pattern =  '(\d+)_decoded.txt'
    r.model_dir = os.path.join(TEST_DIR, "reference")
    r.system_dir = os.path.join(TEST_DIR, "decoded")
    rouge_results = r.convert_and_evaluate('/home/wanglihan/ROUGE/RELEASE-1.5.5/')
    print(rouge_results)

    # a=r.output_to_dict(rouge_results)
    # log_str = ""
    # for x in ["1", "2", "l"]:
    #     log_str += "\nROUGE-%s:\n" % x
    #     for y in ["f_score", "recall", "precision"]:
    #         key = "rouge_%s_%s" % (x, y)
    #         key_cb = key + "_cb"
    #         key_ce = key + "_ce"
    #         val = a[key]
    #         val_cb = a[key_cb]
    #         val_ce = a[key_ce]
    #         log_str += "%s: %.4f with confidence interval (%.4f, %.4f)\n" % (key, val, val_cb, val_ce)
    # print(log_str)



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