import argparse
import pickle
import os
import random
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, Dataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
import model.KVPLM.modeling as modeling
from model.KVPLM.optimization import BertAdam, warmup_linear
from sklearn.metrics import f1_score, roc_auc_score
from chainer_chemistry.dataset.splitters.scaffold_splitter import ScaffoldSplitter
from transformers import AutoTokenizer, AutoModel, BertModel, BertForPreTraining, BertConfig,BertForMaskedLM
from model.KVPLM.smtokenization import SmilesTokenizer


# Note: OldModel is not used in ckpt_KV.pt, because module.ptmodel.bert.embeddings.word_embeddings.weight is not in ckpt_KV.pt; the BigModel use BertModel's forward function.

# Tokenizer of KVPLM is tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')

class OldModel(nn.Module):
    def __init__(self, pt_model):
        super(OldModel, self).__init__()
        self.ptmodel = pt_model
        self.emb = nn.Embedding(390, 768)

    def forward(self, input_ids, attention_mask, token_type_ids):
        '''
        embs = self.ptmodel.bert.embeddings.word_embeddings(input_ids)
        msk = torch.where(input_ids>=30700)
        for k in range(msk[0].shape[0]):
            i = msk[0][k].item()
            j = msk[1][k].item()
            embs[i,j] = self.emb(input_ids[i,j]-30700)
        '''
        msk = (input_ids >= 30700)
        embs = self.emb((input_ids - 30700) * msk)
        return self.ptmodel.bert(inputs_embeds=embs, attention_mask=attention_mask, token_type_ids=token_type_ids)


class BigModel(nn.Module):
    def __init__(self, bert_model, config,multi):
        super(BigModel, self).__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.classifier = []
        for i in range(multi):
            self.classifier.append(nn.Linear(config.hidden_size, 1))
        self.classifier = nn.ModuleList(self.classifier)
        self.multi = multi

    def forward(self, batch):
        pooled = self.bert(batch.tokens, token_type_ids=batch.token_type_ids, attention_mask=batch.attention_mask)['pooler_output']
        encoded = self.dropout(pooled)
        res= [self.classifier[i](encoded) for i in range(self.multi)]
        res=torch.cat(res,1)
        return res


def prepare_kvplm_model(args):
    # if not args.kvplm_language_model: # then do classification
    #     config = modeling.BertConfig.from_json_file(args.config_file)
    #     if config.vocab_size % 8 != 0:
    #         config.vocab_size += 8 - (config.vocab_size % 8)
    #
    #     modeling.ACT2FN["bias_gelu"] = modeling.bias_gelu_training
    #
    #     bert_model0 = BertForPreTraining.from_pretrained('allenai/scibert_scivocab_uncased')
    #     bert_model = OldModel(bert_model0)
    #
    #     if args.init_checkpoint == 'BERT':
    #         con = BertConfig(vocab_size=31090, )
    #         bert_model = BertModel(con)
    #         args.tok = 1
    #         model = BigModel(bert_model, config, args.multi)
    #     elif args.init_checkpoint == 'rxnfp':
    #         bert_model = BertModel.from_pretrained('rxnfp/transformers/bert_mlm_1k_tpl')
    #         args.pth_data += 'rxnfp/'
    #         config.hidden = 256
    #         args.tok = 1
    #         model = BigModel(bert_model, config, args.multi)
    #         args.rx = 1
    #     elif args.init_checkpoint is None:
    #         args.tok = 1
    #         model = BigModel(bert_model0.bert, config, args.multi)
    #     else:
    #         pt = torch.load(args.init_checkpoint)
    #         if 'module.ptmodel.bert.embeddings.word_embeddings.weight' in pt:
    #             pretrained_dict = {k[7:]: v for k, v in pt.items()}
    #             args.tok = 0
    #             bert_model.load_state_dict(pretrained_dict, strict=False)
    #             model = BigModel(bert_model, config, args.multi)
    #         elif 'bert.embeddings.word_embeddings.weight' in pt:
    #             # pretrained_dict = {k[5:]: v for k, v in pt.items()}
    #             args.tok = 1
    #             bert_model0.load_state_dict(pt, strict=True)
    #             model = BigModel(bert_model0.bert, config, args.multi)
    #         else:
    #             raise ValueError('init checkpoint not supported yet')
    # else:
    config = modeling.BertConfig.from_json_file(args.config_file)
    if config.vocab_size % 8 != 0:
        config.vocab_size += 8 - (config.vocab_size % 8)

    modeling.ACT2FN["bias_gelu"] = modeling.bias_gelu_training

    bert_model0 = BertForMaskedLM.from_pretrained('allenai/scibert_scivocab_uncased')
    # bert_model = OldModel(bert_model0)

    if args.init_checkpoint == 'BERT':
        con = BertConfig(vocab_size=31090, )
        model = BertModel(con)
        # args.tok = 1
        # model = BigModel(bert_model, config, args.multi)
    elif args.init_checkpoint == 'rxnfp':
        model = BertModel.from_pretrained('rxnfp/transformers/bert_mlm_1k_tpl')
        # args.pth_data += 'rxnfp/'
        # config.hidden = 256
        # args.tok = 1
        # model = BigModel(bert_model, config, args.multi)
        # args.rx = 1
    elif args.init_checkpoint is None:
        con = BertConfig(vocab_size=31090, )
        model = BertModel(con)
    else:
        if torch.cuda.is_available():
            pt = torch.load(args.init_checkpoint)
        else:
            pt = torch.load(args.init_checkpoint, map_location=torch.device('cpu'))
       
        if 'bert.embeddings.word_embeddings.weight' in pt:
            # pretrained_dict = {k[5:]: v for k, v in pt.items()}
            # args.tok = 1
            missing_keys,unexpected_keys =bert_model0.load_state_dict(pt, strict=False)
            print('missing keys: ',missing_keys)
            print('unexpected keys: ',unexpected_keys)
            model=bert_model0
        else:
            raise ValueError('init checkpoint not supported yet')
    return model




