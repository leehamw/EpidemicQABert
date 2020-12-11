import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_pretrained_bert import BertModel, BertConfig,BertTokenizer
from model.optimizers import Optimizer
import json

INI = 1e-2
def build_optim(args, model, checkpoint):


    if checkpoint is not None:
        optim = checkpoint['optim'][0]
        saved_optimizer_state_dict = optim.optimizer.state_dict()
        optim.optimizer.load_state_dict(saved_optimizer_state_dict)
        if args.visible_gpus != '-1':
            for state in optim.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        if (optim.method == 'adam') and (len(optim.optimizer.state) < 1):
            raise RuntimeError(
                "Error: loaded Adam optimizer from existing model" +
                " but optimizer state is empty")

    else:
        optim = Optimizer(
            args.optim, args.lr, args.max_grad_norm,
            beta1=args.beta1, beta2=args.beta2,
            decay_method='noam',
            warmup_steps=args.warmup_steps)

    optim.set_parameters(list(model.named_parameters()))


    return optim

class Bert(nn.Module):
    def __init__(self,finetune=False):
        super(Bert,self).__init__()

        self.model=BertModel.from_pretrained('./ERINE_pretrain')
        self.tokenizer=BertTokenizer.from_pretrained('./ERINE_pretrain')
        self.finetune=finetune
        if(self.finetune):
            for param in self.model.parameters():
                param.requires_grad = True

    def forward(self, input, seg, mask):

        encoded_layers, pooled =  self.model(input, seg, attention_mask=mask)
        token_embeddings = torch.stack(encoded_layers, dim=0)
        token_vector = torch.sum(token_embeddings,dim=0)
        token_vector = token_vector.permute(1, 0, 2)

        # else:
        #     self.eval()
        #     with torch.no_grad():
        #         encoded_layers, _ = self.model(input, seg, attention_mask=mask)
        #         token_embeddings = torch.stack(encoded_layers, dim=0)
        #         token_vector = torch.sum(token_embeddings[-4:], dim=0)
        #         token_vector = token_vector.permute(1, 0, 2)
        return token_vector


class BertMatcher(nn.Module):
    def __init__(self, finetune, hidden_dim, max_pos):
        super(BertMatcher,self).__init__()

        self.bert=Bert(finetune)
        self.match_layer=nn.Linear(hidden_dim,1, bias=False)
        self.classifier=torch.nn.Sigmoid()
        # init.xavier_normal_(self.match_layer)

        if (max_pos > 512):
            my_pos_embeddings = nn.Embedding(max_pos, self.bert.model.config.hidden_size)
            my_pos_embeddings.weight.data[:512] = self.bert.model.embeddings.position_embeddings.weight.data
            my_pos_embeddings.weight.data[512:] = self.bert.model.embeddings.position_embeddings.weight.data[-1][None,
                                                  :].repeat(max_pos - 512, 1)
            self.bert.model.embeddings.position_embeddings = my_pos_embeddings


    def forward(self, src, segs, mask_src):
        token_vec=self.bert(src, segs, mask_src)
        input_vec=token_vec[0]
        output=self.match_layer(input_vec)
        scores=self.classifier(output)
        return scores






























if __name__ == '__main__':
    # f= open('../val.txt','r')
    # m=f.readlines()
    # for m1 in m:
    #    dict=json.loads(m1)
    #    print(dict.keys())
    #    print(type(dict['article_text']))
    #    print(dict['labels'])
    #    for k in dict['abstract_text']:
    #        print(k)
    #    # print(dict['new_docid'])
    #    print(m1)
    #    break


    # bert_path = '../ERINE_pretrain'
    #
    # tokenizer = BertTokenizer.from_pretrained(bert_path)
    # text='我是一头小毛驴，。'
    # marked_text = "[CLS] " + text+" [SEP]"
    # tokenized_text = tokenizer.tokenize(marked_text)
    # print(tokenized_text)
    # indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    # print(indexed_tokens)
    # seg=[]
    #
    # seg.append([1]*(indexed_tokens.index(2)+1)+[0]*(len(indexed_tokens)-1-indexed_tokens.index(2)))
    # print(seg)

    tokenizer = BertTokenizer.from_pretrained('../bert-base-uncased')
    text = '[UNK]'.split()
    print(type(text))
    arr=[]
    for m in text:
        arr.append(tokenizer.tokenize(m)[0])
        print(type(tokenizer.tokenize(m)))
    print(arr)
    # marked_text = "[CLS] " + text + " [SEP]"

    # tokenized_text = tokenizer.tokenize(marked_text)
    # print(tokenized_text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(arr)
    print(indexed_tokens)

    text2=tokenizer.tokenize("[CLS] " +'[UNK]'+ " [SEP]")
    print('ddd {}'.format(text2))

    # in_2=tokenizer.convert_tokens_to_ids(text2)
    # in_2=in_2+[0]*(len(tokenized_text)-len(in_2))
    # print(in_2)
    # segments_ids = [1] * len(tokenized_text)
    # sg_2=[1]* len(tokenized_text)


    gg=[indexed_tokens,indexed_tokens]
    gs=[segments_ids,segments_ids]
    # for i ,w in enumerate(gg):
    #     id_arr[i]=w[:max(len(tokenized_text,len(in_2)))]
    token_type = torch.LongTensor
    mask_type = torch.ByteTensor
    segment_type =torch.LongTensor
    tensor_shape = (2, 11)
    token_tensor = token_type(*tensor_shape)
    segments_tensor=token_type(*tensor_shape)

    for i, ids in enumerate(gg):
        token_tensor[i, :] = token_type(ids)

    mask_src=torch.ByteTensor(2,11)
    mask_src.fill_(1)
    mask_src[1][6:].fill_(0)

    #
    #
    # # mask_src=~(tokens_tensor==0)
    # # print(mask_src)
    for i, ids in enumerate(gs):
        segments_tensor[i, :] = token_type(ids)
    # segments_tensors = torch.LongTensor(gs)

    # model = BertModel.from_pretrained(bert_path)

    encoded_layers, _= model( token_tensor, segments_tensor,attention_mask=mask_src)
    token_embeddings = torch.stack(encoded_layers, dim=0)
    # # token_embeddings = encoded_layers
    print(token_embeddings.size())
    token_embeddings = torch.sum(token_embeddings[-4:], dim=0)
    print(token_embeddings.size())
    token_embeddings = token_embeddings.permute(1, 0, 2)
    print(token_embeddings[0].size())
    # # token_vecs_sum = []
    # # for token in token_embeddings:
    # #     sum_vec=torch.sum(token[-4:],dim=0)
    # #     token_vecs_sum=[]
    # bert_config = BertConfig(model.config.vocab_size, num_hidden_layers=12)
    # model.model=BertModel(bert_config)
    # with torch.no_grad():
    #     encoded_layers, _ = model(tokens_tensor, segments_tensors,attention_mask=mask_src)
    #     token_embeddings = torch.stack(encoded_layers, dim=0)
    #     print(token_embeddings.size())
    # #sentence vector
    # '''
    # tokens_tensor = tokens_tensor.to('cuda')
    # segments_tensors = segments_tensors.to('cuda')
    #
    # '''
