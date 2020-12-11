import random
import torch
import torch.multiprocessing as mp
from toolz.sandbox import unzip
from pytorch_pretrained_bert import BertTokenizer
from cytoolz import curry, concat

def coll_fn(data):

    label_lists, concat_text_lists= unzip(data)
    # print(type(label_lists))
    # print(type(concat_text_lists))
    labels = list(label_lists)
    concat_texts = list(concat_text_lists)

    assert all(concat_texts)
    return labels, concat_texts


def tokenize(max_pos,concat_texts):
    tokenizer = BertTokenizer.from_pretrained('./ERINE_pretrain')
    tokenized_texts=[tokenizer.tokenize(concat_text) for concat_text in concat_texts]
    indexed_tokens = [tokenizer.convert_tokens_to_ids(tokenized_text) for tokenized_text in tokenized_texts]
    indexed_tokens=[indexed_token[:max_pos] for indexed_token in indexed_tokens]

    return indexed_tokens


@curry
def prepro_fn(max_pos, batch):

    labels, concat_texts = batch

    assert len(labels) == len(concat_texts)
    # indexed_tokens=tokenize(max_pos, concat_texts)
    batch=list(zip(labels,concat_texts))

    return batch

@curry
def pad_batch_tensorize(indexed_tokens, cuda=True):

   #bert 输入
   token_type = torch.cuda.LongTensor if cuda else torch.LongTensor
   mask_type = torch.cuda.ByteTensor if cuda else torch.ByteTensor
   segment_type=torch.cuda.LongTensor if cuda else torch.LongTensor


   batch_size = len(indexed_tokens)
   max_len = max(len(ids) for ids in indexed_tokens)

   tensor_shape = (batch_size, max_len)
   token_tensor=token_type(*tensor_shape)
   mask_tensor=mask_type(*tensor_shape)
   segment_tensor=segment_type(*tensor_shape)



   indexed_tokens_after_pad=[indexed_token+[0]*(max_len-len(indexed_token)) for indexed_token in indexed_tokens]
   for i, ids in enumerate(indexed_tokens_after_pad):
       token_tensor[i,:]=token_type(ids)
   segment_ids_after_pad=[[1]*(indexed_token.index(102)+1) +[0]*(max_len-1-indexed_token.index(102)) if 102 in indexed_token else [1] * max_len for indexed_token in indexed_tokens]
   for s, sds in enumerate(segment_ids_after_pad):
       segment_tensor[s,:]=segment_type(sds)
   mask_after_pad=[[1]*len(indexed_token)+[0]*(max_len-len(indexed_token)) for indexed_token in indexed_tokens]
   for m, mds in enumerate(mask_after_pad):
       mask_tensor[m,:]=mask_type(mds)
   return token_tensor, segment_tensor, mask_tensor



@curry
def batchify_fn(data, cuda=True):
    labels, indexed_tokens = map(list, unzip(data))
    token_tensor, segment_tensor, mask_tensor=pad_batch_tensorize(indexed_tokens, cuda)
    fw_args = (token_tensor, segment_tensor, mask_tensor)
    assert len(labels) ==len(indexed_tokens)

    label_tensor = torch.tensor(labels)
    # label_tensor=torch.unsqueeze(label_tensor,dim=1)
    label_tensor=label_tensor.float()

    loss_args = label_tensor.cuda() if cuda else label_tensor
    return fw_args, loss_args



def _batch2q(loader,prepro,  q, single_run=True):
    epoch = 0
    while True:
        for batch in loader:
            q.put(prepro(batch))
        if single_run:
            break
        epoch += 1
        q.put(epoch)
    q.put(None)

class BucketedGenerater(object):
    def __init__(self, loader, sort_key, prepro, batchify, single_run=True, queue_size=8, fork=True):
        self._loader=loader
        self._prepro=prepro #处理一下数组，做一下截断之类的
        self._batchify=batchify #做一下padding
        self._single_run = single_run
        self._sort_key=sort_key
        if fork:
            ctx = mp.get_context('forkserver')
            self._queue = ctx.Queue(queue_size)
        else:
            # for easier debugging
            self._queue = None
        self._process = None

    def __call__(self,batch_size: int):
        def get_batches(hyper_batch):
            indexes=list(range(0, len(hyper_batch), batch_size))
            if not self._single_run:
                random.shuffle(hyper_batch)
                random.shuffle(indexes)
            hyper_batch.sort(key=self._sort_key)
            for i in indexes:
                batch=self._batchify(hyper_batch[i:i+batch_size])
                yield batch

        if self._queue is not None:
            ctx=mp.get_context('forkserver')
            self._process = ctx.Process(
                target=_batch2q,
                args=(self._loader, self._prepro,
                      self._queue, self._single_run)
            )
            self._process.start()
            while True:
                d = self._queue.get()
                if d is None:
                    break
                if isinstance(d, int):
                    print('\nepoch {} done'.format(d))
                    continue
                yield from get_batches(d)
            self._process.join()
        else:
            i = 0
            while True:
                for batch in self._loader:
                    yield from get_batches(self._prepro(batch))
                if self._single_run:
                    break
                i += 1
                print('\nepoch {} done'.format(i))

    def terminate(self):
        if self._process is not None:
            self._process.terminate()
            self._process.join()



