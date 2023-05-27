import argparse
import random
import numpy as np
import torch
from torch.autograd import Variable
from utils.GraphAug import drop_nodes, permute_edges, subgraph, mask_nodes
from copy import deepcopy
from tqdm import tqdm
from model.contrastive_gin import GINSimclr
from data_provider.match_dataset import GINMatchDataset
from data_provider.sent_dataset import GINSentDataset
import torch_geometric
from transformers import BertTokenizer
import torch.optim as optim
from optimization import BertAdam, warmup_linear
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, Dataset)
import pubchempy as pcp

def prepare_model_and_optimizer(args, device):
    model = GINSimclr.load_from_checkpoint(args.init_checkpoint)
    model.to(device)
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [
                p for n, p in param_optimizer
                if not any(nd in n for nd in no_decay)
            ],
            'weight_decay': 0.01
        },
        {
            'params': [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            'weight_decay': 0.0
        },
    ]
    optimizer = BertAdam(
            optimizer_grouped_parameters,
            weight_decay=args.weight_decay,
            lr=args.lr,
            warmup=args.warmup,
            t_total=args.total_steps,
            )
    return model,optimizer

def augment(data, graph_aug):
    
    if graph_aug == 'dnodes':
        print('1111111')
        data_aug = drop_nodes(deepcopy(data))
    elif graph_aug == 'pedges':
        data_aug = permute_edges(deepcopy(data))
    elif graph_aug == 'subgraph':
        data_aug = subgraph(deepcopy(data))
    elif graph_aug == 'mask_nodes':
        data_aug = mask_nodes(deepcopy(data))
    elif graph_aug == 'random2':  # choose one from two augmentations
        n = np.random.randint(2)
        if n == 0:
            data_aug = drop_nodes(deepcopy(data))
        elif n == 1:
            data_aug = subgraph(deepcopy(data))
        else:
            print('sample error')
            assert False
    elif graph_aug == 'random3':  # choose one from three augmentations
        n = np.random.randint(3)
        if n == 0:
            data_aug = drop_nodes(deepcopy(data))
        elif n == 1:
            data_aug = permute_edges(deepcopy(data))
        elif n == 2:
            data_aug = subgraph(deepcopy(data))
        else:
            print('sample error')
            assert False
    elif graph_aug == 'random4':  # choose one from four augmentations
        n = np.random.randint(4)
        if n == 0:
            data_aug = drop_nodes(deepcopy(data))
        elif n == 1:
            data_aug = permute_edges(deepcopy(data))
        elif n == 2:
            data_aug = subgraph(deepcopy(data))
        elif n == 3:
            data_aug = mask_nodes(deepcopy(data))
        else:
            print('sample error')
            assert False
    else:
        data_aug = deepcopy(data)
        data_aug.x = torch.ones((data.edge_index.max()+1, 1))
    
    return data_aug

def main(args):
    #print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    device = torch.device('cuda')
    model, optimizer = prepare_model_and_optimizer(args, device)

    # TrainSet = GINMatchDataset(args.pth_train + '/', args)
    # DevSet = GINMatchDataset(args.pth_dev + '/', args)
    TestSet = GINMatchDataset(args.pth_test + '/', args)
    #train_sampler = RandomSampler(TrainSet)
    # train_dataloader = torch_geometric.loader.DataLoader(TrainSet, sampler=train_sampler,
    #                               batch_size=args.batch_size,
    #                               num_workers=0, pin_memory=True, drop_last=True)
    # dev_dataloader = torch_geometric.loader.DataLoader(DevSet, shuffle=False,
    #                               batch_size=args.batch_size,
    #                               num_workers=0, pin_memory=True, drop_last=True)
    test_dataloader = torch_geometric.loader.DataLoader(TestSet, shuffle=False,
                                  batch_size=args.batch_size,
                                  num_workers=4, pin_memory=True, drop_last=False)#True
    if args.if_test == 4:
        model.load_state_dict(torch.load(args.output))
        graph_rep = torch.from_numpy(np.load('output/graph_rep.npy'))
        graph_rep = graph_rep.cuda()
        while True:
            print('*********************************************')
            des = input('input description:')
            des.strip('\n')
            #des = 'It is a diamine and an alkanethiol. It derives from a cysteamine. '
            #print(des)
            tokenizer = BertTokenizer.from_pretrained('/hy-tmp/bert_pretrained/')
            sentence_token = tokenizer(text=des,
                                    truncation=True,
                                    padding='max_length',
                                    add_special_tokens=False,
                                    max_length=args.text_max_len,
                                    return_tensors='pt',
                                    return_attention_mask=True)
            text = sentence_token['input_ids']  # [176,398,1007,0,0,0]
            mask = sentence_token['attention_mask']  # [1,1,1,0,0,0]
            # print(text)
            # print(mask)
            text = text.cuda()
            mask = mask.cuda()
            text_rep = model.text_encoder(text, mask)
            text_rep = model.text_proj_head(text_rep)
            #print(graph_rep.shape)
            #print(text_rep.shape)

            graph_len = graph_rep.shape[0]
            text_len = text_rep.shape[0]

            score_tmp = torch.zeros(text_len, graph_len)
            for i in range(text_len):
                score_tmp[i] = torch.cosine_similarity(text_rep[i], graph_rep, dim=-1)
                score_tmp[i], indices = torch.sort(score_tmp[i], descending=True)
                #print(score_tmp[i][:10])
                #print(indices[:10])

            idx = np.load('output/idx.npy')
            print(idx.shape)
            des = open('/hy-tmp/KV-PLM/Ret/align_des_filt3.txt').readlines()
            sms = open('/hy-tmp/KV-PLM/Ret/align_smiles.txt').readlines()
            for ii,i  in enumerate(indices[:10]):
                print(ii+1)
                print(idx[i].astype(int))
                x1 = sms[idx[i].astype(int)-1]
                x2 = des[idx[i].astype(int)-1]
                # p = pcp.get_compounds(x1, 'smiles')[0]
                # # c = pcp.Compound.from_cid(12)
                # n = p.synonyms[0]
                print(x1)
                #print(n)
                print(x2)
    
    if args.if_test == 5:
        model.load_state_dict(torch.load(args.output))
        text_rep = None
        graph_rep = torch.from_numpy(np.load('output/text_rep.npy'))
        #graph_rep = graph_rep.cuda()
        while True:
            print('*********************************************')
            des = input('input graph:')
            des.strip('\n')
            rep = torch.from_numpy(np.load('output/graph_rep.npy'))
            idx = np.load('output/idx.npy')
            #des = des.int()
            des = int(des)
            for i in range(len(idx)):
                #print(i)
                #print(idx[i])
                if idx[i] == des:
                    #print(i)
                    text_rep = rep[i]
                    break
            text_rep = text_rep.view(1, text_rep.shape[0])
            print(text_rep.shape)
            print(graph_rep.shape)
            #pth = '/hy-tmp/kv_data/test/graph/'
            #pth = pth + f'graph_{des}.pt'
            #data_graph = torch.load(pth)
            #data_aug = augment(data_graph, 'dnodes')
            #data_aug = data_aug.to('cuda')
            #data_aug = data_aug.unsequeeze(0)
            #des = 'It is a diamine and an alkanethiol. It derives from a cysteamine. '
            #print(des)
            # tokenizer = BertTokenizer.from_pretrained('/hy-tmp/bert_pretrained/')
            # sentence_token = tokenizer(text=des,
            #                         truncation=True,
            #                         padding='max_length',
            #                         add_special_tokens=False,
            #                         max_length=args.text_max_len,
            #                         return_tensors='pt',
            #                         return_attention_mask=True)
            # text = sentence_token['input_ids']  # [176,398,1007,0,0,0]
            # mask = sentence_token['attention_mask']  # [1,1,1,0,0,0]
            # # print(text)
            # # print(mask)
            # text = text.cuda()
            # mask = mask.cuda()
            # text_rep = model.graph_encoder(data_aug)
            # text_rep = model.graph_proj_head(data_aug)
            #print(graph_rep.shape)
            #print(text_rep.shape)

            graph_len = graph_rep.shape[0]
            text_len = text_rep.shape[0]

            score_tmp = torch.zeros(text_len, graph_len)
            for i in range(text_len):
                score_tmp[i] = torch.cosine_similarity(text_rep[i], graph_rep, dim=-1)
                score_tmp[i], indices = torch.sort(score_tmp[i], descending=True)
                #print(score_tmp[i][:10])
                #print(indices[:10])

            
            des = open('/hy-tmp/KV-PLM/Ret/align_des_filt3.txt').readlines()
            sms = open('/hy-tmp/KV-PLM/Ret/align_smiles.txt').readlines()
            for ii,i  in enumerate(indices[:10]):
                print(ii+1)
                print(idx[i].astype(int))
                x1 = sms[idx[i].astype(int)-1]
                x2 = des[idx[i].astype(int)-1]
                # p = pcp.get_compounds(x1, 'smiles')[0]
                # # c = pcp.Compound.from_cid(12)
                # n = p.synonyms[0]
                print(x1)
                #print(n)
                print(x2)
        return 

    
    # if 'sci' in args.init_checkpoint:
    #     args.init_checkpoint = 'sci'
    # else:
    #     args.init_checkpoint = 'kv'
    # with open(f'log/pt={args.init_checkpoint}-epoch={args.epoch}-seed={args.seed}.txt','w') as f:
    #     f.writelines(str(acc1) + '\n' + str(acc2))

def parse_args(parser=argparse.ArgumentParser()):
    parser.add_argument("--init_checkpoint", default="/hy-tmp/MoMu-K.ckpt", type=str,)
    parser.add_argument("--output", default='finetune_save/sent_MoMu-K_73.pt', type=str,)
    parser.add_argument("--data_type", default=1, type=int) # 0-para, 1-sent
    parser.add_argument("--if_test", default=4, type=int)
    parser.add_argument("--if_zeroshot", default=0, type=int)
    parser.add_argument("--pth_train", default='/hy-tmp/kv_data/train', type=str,)
    parser.add_argument("--pth_dev", default='/hy-tmp/kv_data/dev', type=str,)
    parser.add_argument("--pth_test", default='/hy-tmp/kv_data/test', type=str,)
    parser.add_argument("--weight_decay", default=0, type=float,)
    parser.add_argument("--lr", default=5e-5, type=float,)#4
    parser.add_argument("--warmup", default=0.2, type=float,)
    parser.add_argument("--total_steps", default=5000, type=int,)#3000
    parser.add_argument("--batch_size", default=64, type=int,)
    parser.add_argument("--epoch", default=30, type=int,)
    parser.add_argument("--seed", default=73, type=int,)#73 99 108
    parser.add_argument("--graph_aug", default='dnodes', type=str,)
    parser.add_argument("--text_max_len", default=128, type=int,)
    parser.add_argument("--margin", default=0.2, type=int,)
    
    
    args = parser.parse_args()
    # ckpt = ''
    # if 'sci' in args.init_checkpoint:
    #         ckpt = 'sci'
    # else:
    #     ckpt = 'kv'
    # args.output = f'finetune_save/pt={ckpt}-seed={args.seed}.pt'
    return args

if __name__ == "__main__":
    main(parse_args())