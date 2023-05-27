import argparse
import random
import numpy as np
import torch
from torch.autograd import Variable
from tqdm import tqdm
from model.contrastive_gin import GINSimclr
from data_provider.match_dataset import GINMatchDataset
from data_provider.sent_dataset import GINSentDataset
import torch_geometric
from optimization import BertAdam, warmup_linear
from torch.utils.data import RandomSampler
import os

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

    # pt = model.state_dict()
    # for k in pt.keys():
    #     print(k)

    # print(pt['graph_encoder.gnns.1.mlp.0.weight'])
    # print(pt['text_encoder.main_model.encoder.layer.1.attention.self.key.weight'])
    # print(pt['graph_proj_head.0.weight'])
    # print(pt['text_proj_head.0.weight'])

    return model,optimizer

def Eval(model, dataloader, device, args):
    
    model.eval()
    with torch.no_grad():
        acc1 = 0
        acc2 = 0
        allcnt = 0
        graph_rep_total = None
        text_rep_total = None
        for batch in tqdm(dataloader):
            aug, text, mask = batch
            aug = aug.to(device)
            # print(aug)
            # text = text.to(device)
            # mask = mask.to(device)
            text = text.to(device)
            # print(text)
            mask = mask.to(device)
            # print(mask)
            graph_rep = model.graph_encoder(aug)
            graph_rep = model.graph_proj_head(graph_rep)

            # print('graph')
            # print(graph_rep[:,0])
            text_rep = model.text_encoder(text, mask)
            text_rep = model.text_proj_head(text_rep)
            # print('text')
            # print(text_rep[:,0])
            scores1 = torch.cosine_similarity(graph_rep.unsqueeze(1).expand(graph_rep.shape[0], graph_rep.shape[0], graph_rep.shape[1]), text_rep.unsqueeze(0).expand(text_rep.shape[0], text_rep.shape[0], text_rep.shape[1]), dim=-1)
            scores2 = torch.cosine_similarity(text_rep.unsqueeze(1).expand(text_rep.shape[0], text_rep.shape[0], text_rep.shape[1]), graph_rep.unsqueeze(0).expand(graph_rep.shape[0], graph_rep.shape[0], graph_rep.shape[1]), dim=-1)
            # print(scores1)
            # print(scores2)
            argm1 = torch.argmax(scores1, axis=1)
            argm2 = torch.argmax(scores2, axis=1)

            acc1 += sum((argm1==torch.arange(argm1.shape[0]).to(device)).int()).item()
            acc2 += sum((argm2==torch.arange(argm2.shape[0]).to(device)).int()).item()

            allcnt += argm1.shape[0]

            if graph_rep_total is None or text_rep_total is None:
                graph_rep_total = graph_rep
                text_rep_total = text_rep
            else:
                graph_rep_total = torch.cat((graph_rep_total, graph_rep), axis=0)
                text_rep_total = torch.cat((text_rep_total, text_rep), axis=0)
     
    if args.if_test == 2: #save rep to caculate rec@20
        if not os.path.exists('output'):
            os.mkdir('output')
        np.save('output/graph_rep.npy', graph_rep_total.cpu())
        np.save('output/text_rep.npy', text_rep_total.cpu())

    return acc1/allcnt, acc2/allcnt

# get every sentence's rep
def CalSent(model, dataloader, device, args): 
    model.eval()
    with torch.no_grad():
        text_rep_total = None
        for batch in tqdm(dataloader):
            text, mask = batch
            text = text.to(device)
            mask = mask.to(device)
            text_rep = model.text_encoder(text, mask)
            text_rep = model.text_proj_head(text_rep)

            if text_rep_total is None:
                text_rep_total = text_rep
            else:
                text_rep_total = torch.cat((text_rep_total, text_rep), axis=0)

    if args.if_test == 2:
        np.save('output/text_rep.npy', text_rep_total.cpu())

def Contra_Loss(logits_des, logits_smi, margin, device):
    scores = torch.cosine_similarity(logits_smi.unsqueeze(1).expand(logits_smi.shape[0], logits_smi.shape[0], logits_smi.shape[1]), logits_des.unsqueeze(0).expand(logits_des.shape[0], logits_des.shape[0], logits_des.shape[1]), dim=-1)
    diagonal = scores.diag().view(logits_smi.size(0), 1)
    d1 = diagonal.expand_as(scores)
    d2 = diagonal.t().expand_as(scores)
    
    cost_des = (margin + scores - d1).clamp(min=0)
    cost_smi = (margin + scores - d2).clamp(min=0)

    # clear diagonals
    mask = torch.eye(scores.size(0)) > .5
    I = Variable(mask)
    if torch.cuda.is_available():
        I = I.to(device)
    cost_des = cost_des.masked_fill_(I, 0)
    cost_smi = cost_smi.masked_fill_(I, 0)

    # keep the maximum violating negative for each query
    #if self.max_violation:
    cost_des = cost_des.max(1)[0]
    cost_smi = cost_smi.max(0)[0]

    return cost_des.sum() + cost_smi.sum()

def main(args):
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    device = torch.device(f'cuda:{args.device}')
    model, optimizer = prepare_model_and_optimizer(args, device)

    TrainSet = GINMatchDataset(args.pth_train + '/', args)
    DevSet = GINMatchDataset(args.pth_dev + '/', args)
    TestSet = GINMatchDataset(args.pth_test + '/', args)
    train_sampler = RandomSampler(TrainSet)
    train_dataloader = torch_geometric.loader.DataLoader(TrainSet, sampler=train_sampler,
                                  batch_size=args.batch_size,
                                  num_workers=4, pin_memory=True, drop_last=True)
    dev_dataloader = torch_geometric.loader.DataLoader(DevSet, shuffle=False,
                                  batch_size=args.batch_size,
                                  num_workers=4, pin_memory=True, drop_last=True)
    test_dataloader = torch_geometric.loader.DataLoader(TestSet, shuffle=False,
                                  batch_size=args.batch_size,
                                  num_workers=4, pin_memory=True, drop_last=False)#True
    global_step = 0
    tag = True
    best_acc = 0

    if args.if_test == 1: #calculate Acc only
        if args.if_zeroshot == 0: #finetuned
            model.load_state_dict(torch.load(args.output))
        acc1, acc2 = Eval(model, test_dataloader, device, args)
        print('Test Acc1:', acc1)
        print('Test Acc2:', acc2)
        return

    elif args.if_test == 2: #calculate Rec
        if args.data_type == 0: # para-level
            if args.if_zeroshot == 0: #finetuned
                model.load_state_dict(torch.load(args.output))
            acc1, acc2 = Eval(model, test_dataloader, device, args)
            print('Test Acc1:', acc1)
            print('Test Acc2:', acc2)
            graph_rep = torch.from_numpy(np.load('output/graph_rep.npy'))
            text_rep = torch.from_numpy(np.load('output/text_rep.npy'))
            graph_len = graph_rep.shape[0]
            text_len = text_rep.shape[0]
            score1 = torch.zeros(graph_len, graph_len)
            for i in range(graph_len):
                score1[i] = torch.cosine_similarity(graph_rep[i], text_rep, dim=-1)
            rec1 = []
            for i in range(graph_len):
                a,idx = torch.sort(score1[:,i])
                for j in range(graph_len):
                    if idx[-1-j]==i:
                        rec1.append(j)
                        break
            print(f'Rec@20 1: {sum( (np.array(rec1)<20).astype(int) ) / graph_len}')
            score2 = torch.zeros(graph_len, graph_len)
            for i in range(graph_len):
                score2[i] = torch.cosine_similarity(text_rep[i], graph_rep, dim=-1)
            rec2 = []
            for i in range(graph_len):
                a,idx = torch.sort(score2[:,i])
                for j in range(graph_len):
                    if idx[-1-j]==i:
                        rec2.append(j)
                        break
            print(f'Rec@20 2: {sum( (np.array(rec2)<20).astype(int) ) / graph_len}')
        
        else: #sent-level
            if args.if_zeroshot == 0: #finetuned
                model.load_state_dict(torch.load(args.output))
            acc1, acc2 = Eval(model, test_dataloader, device, args)
            print('Test Acc1:', acc1)
            print('Test Acc2:', acc2)
            graph_rep = torch.from_numpy(np.load('output/graph_rep.npy'))
            SentSet = GINSentDataset(args.pth_test + '/', args)
            sent_dataloader = torch_geometric.loader.DataLoader(SentSet, shuffle=False,
                                  batch_size=args.batch_size,
                                  num_workers=4, pin_memory=True, drop_last=False)#True
            CalSent(model, sent_dataloader, device, args)
            graph_rep = torch.from_numpy(np.load('output/graph_rep.npy'))
            text_rep = torch.from_numpy(np.load('output/text_rep.npy'))
            cor = np.load('output/cor.npy')
            
            graph_len = graph_rep.shape[0]
            text_len = text_rep.shape[0]

            score1 = torch.zeros(graph_len, graph_len)
            score2 = torch.zeros(graph_len, graph_len)

            for i in range(graph_len):
                score = torch.cosine_similarity(graph_rep[i], text_rep, dim=-1)
                for j in range(graph_len):
                    total = 0
                    for k in range(cor[j], cor[j+1]):
                        total+=(score[k]/(cor[j+1]-cor[j]))
                    score1[i,j] = total
                    #score1[i,j] = sum(score[cor[j]:cor[j+1]])/(cor[j+1]-cor[j])
            rec1 = []
            for i in range(graph_len):
                a,idx = torch.sort(score1[:,i])
                for j in range(graph_len):
                    if idx[-1-j]==i:
                        rec1.append(j)
                        break
            print(f'Rec@20 1: {sum( (np.array(rec1)<20).astype(int) ) / graph_len}')

            score_tmp = torch.zeros(text_len, graph_len)
            for i in range(text_len):
                score_tmp[i] = torch.cosine_similarity(text_rep[i], graph_rep, dim=-1)
            score_tmp = torch.t(score_tmp)

            for i in range(graph_len):
                for j in range(graph_len):
                    total = 0
                    for k in range(cor[j], cor[j+1]):
                        total+=(score_tmp[i][k]/(cor[j+1]-cor[j]))
                    score2[i,j] = total
                    #score2[i,j] = sum(score_tmp[i][cor[j]:cor[j+1]])/(cor[j+1]-cor[j])
            score2 = torch.t(score2)

            rec2 = []
            for i in range(graph_len):
                a,idx = torch.sort(score2[:,i])
                for j in range(graph_len):
                    if idx[-1-j]==i:
                        rec2.append(j)
                        break
            print(f'Rec@20 2: {sum( (np.array(rec2)<20).astype(int) ) / graph_len}')
        
        return

    for epoch in range(args.epoch):
        if tag==False:
            break
        acc1, acc2 = Eval(model, dev_dataloader, device, args)
        print('Epoch:', epoch, ', DevAcc1:', acc1)
        print('Epoch:', epoch, ', DevAcc2:', acc2)
        if acc1>best_acc:
            best_acc = acc1
            torch.save(model.state_dict(), args.output)
            print('Save checkpoint ', global_step)
        acc = 0
        allcnt = 0
        sumloss = 0
        model.train()
        for idx,batch in enumerate(tqdm(train_dataloader)):
            aug, text, mask = batch
            aug.to(device)
            text = text.to(device)
            mask = mask.to(device)

            graph_rep = model.graph_encoder(aug)
            graph_rep = model.graph_proj_head(graph_rep)

            text_rep = model.text_encoder(text, mask)
            text_rep = model.text_proj_head(text_rep)

            loss = Contra_Loss(graph_rep, text_rep, args.margin, device)
            scores = text_rep.mm(graph_rep.t())
            argm = torch.argmax(scores, axis=1)
            acc += sum((argm==torch.arange(argm.shape[0]).to(device)).int()).item()
            allcnt += argm.shape[0]
            sumloss += loss.item()
            loss.backward()
            #if idx%4==1:
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
            if global_step>args.total_steps:
                tag = False
                break
        optimizer.step()
        optimizer.zero_grad()
        print('Epoch:', epoch, ', Acc:', acc/allcnt, ', Loss:', sumloss/allcnt)
    acc1, acc2 = Eval(model, dev_dataloader, device, args)
    print('Epoch:', args.epoch, ', DevAcc1:', acc1)
    print('Epoch:', args.epoch, ', DevAcc2:', acc2)
    if acc1>best_acc:
        best_acc = acc1
        torch.save(model.state_dict(), args.output)
        print('Save checkpoint ', global_step)
    model.load_state_dict(torch.load(args.output))
    acc1, acc2 = Eval(model, test_dataloader, device, args)
    print('Test Acc1:', acc1)
    print('Test Acc2:', acc2)

def parse_args(parser=argparse.ArgumentParser()):
    parser.add_argument("--device", default="0", type=str,)
    parser.add_argument("--init_checkpoint", default="all_checkpoints/MoMu-S.ckpt", type=str,)
    parser.add_argument("--output", default='finetune_save/sent_MoMu-S_73.pt', type=str,)
    parser.add_argument("--data_type", default=0, type=int) # 0-para, 1-sent
    parser.add_argument("--if_test", default=1, type=int)
    parser.add_argument("--if_zeroshot", default=1, type=int)
    parser.add_argument("--pth_train", default='data/kv_data/train', type=str,)
    parser.add_argument("--pth_dev", default='data/kv_data/dev', type=str,)
    parser.add_argument("--pth_test", default='data/phy_data', type=str,)
    parser.add_argument("--weight_decay", default=0, type=float,)
    parser.add_argument("--lr", default=5e-5, type=float,)#4
    parser.add_argument("--warmup", default=0.2, type=float,)
    parser.add_argument("--total_steps", default=5000, type=int,)#3000
    parser.add_argument("--batch_size", default=64, type=int,)
    parser.add_argument("--epoch", default=30, type=int,)
    parser.add_argument("--seed", default=73, type=int,)#73 99 108
    parser.add_argument("--graph_aug", default='noaug', type=str,)
    parser.add_argument("--text_max_len", default=128, type=int,)
    parser.add_argument("--margin", default=0.2, type=int,)
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    main(parse_args())