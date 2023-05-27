from os.path import join
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
# from config import args
from sklearn.metrics import (r2_score,
                             roc_auc_score)
from dataloaders.splitters import random_scaffold_split, random_split, scaffold_split
from torch.utils.data import DataLoader
# from util import get_num_task
import plotly.graph_objects as go
import argparse

import commentjson
# from model import GinT5TransformerForConditionalGeneration
from model import GraphormerModel, GraphormerConfig,GinConfig,Graphormer_version_dict,KVPLMConfig,MoMuConfig,MolT5Config, GalacticaConfig, GPT3Config, get_model
from dataloaders import add_prompt_conditional_generation_transform_single_dict,\
    add_prompt_conditional_generation_transform_sample_dict,\
    GraphTransformer_collator_dict, \
    MoleculeDatasetSplitLabel

from transformers import (
    AutoTokenizer,
    HfArgumentParser,
)
from tqdm import tqdm
import os
import re

os.environ["TOKENIZERS_PARALLELISM"] = "false"

parser = argparse.ArgumentParser()

# about seed and basic info
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--runseed', type=int, default=0)
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--no_cuda',action='store_true')
parser.add_argument('--disable_tqdm',action='store_true')

# about dataset and dataloader
parser.add_argument('--input_data_dir', type=str, default='')
parser.add_argument('--dataset', type=str, default='bace')
parser.add_argument('--num_workers', type=int, default=0)

# about training strategies
parser.add_argument('--split', type=str, default='scaffold')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--grad_accum_step',type=int,default=1)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--lr_scale', type=float, default=1)
parser.add_argument('--decay', type=float, default=0)


parser.add_argument('--graph_pooling', type=str, default='mean')
parser.add_argument('--JK', type=str, default='last')
parser.add_argument('--gnn_lr_scale', type=float, default=1)
parser.add_argument('--model_3d', type=str, default='schnet', choices=['schnet'])

# for AttributeMask
parser.add_argument('--mask_rate', type=float, default=0.15)
parser.add_argument('--mask_edge', type=int, default=0)

# for ContextPred
parser.add_argument('--csize', type=int, default=3)
parser.add_argument('--contextpred_neg_samples', type=int, default=1)

# for SchNet
parser.add_argument('--num_filters', type=int, default=128)
parser.add_argument('--num_interactions', type=int, default=6)
parser.add_argument('--num_gaussians', type=int, default=51)
parser.add_argument('--cutoff', type=float, default=10)
parser.add_argument('--readout', type=str, default='mean', choices=['mean', 'add'])
parser.add_argument('--schnet_lr_scale', type=float, default=1)

# for 2D-3D Contrastive CL
parser.add_argument('--CL_neg_samples', type=int, default=1)
parser.add_argument('--CL_similarity_metric', type=str, default='InfoNCE_dot_prod',
                    choices=['InfoNCE_dot_prod', 'EBM_dot_prod'])
parser.add_argument('--T', type=float, default=0.1)
parser.add_argument('--normalize', dest='normalize', action='store_true')
parser.add_argument('--no_normalize', dest='normalize', action='store_false')
parser.add_argument('--SSL_masking_ratio', type=float, default=0)
# This is for generative SSL.
parser.add_argument('--AE_model', type=str, default='AE', choices=['AE', 'VAE'])
parser.set_defaults(AE_model='AE')

# for 2D-3D AutoEncoder
parser.add_argument('--AE_loss', type=str, default='l2', choices=['l1', 'l2', 'cosine'])
parser.add_argument('--detach_target', dest='detach_target', action='store_true')
parser.add_argument('--no_detach_target', dest='detach_target', action='store_false')
parser.set_defaults(detach_target=True)

# for 2D-3D Variational AutoEncoder
parser.add_argument('--beta', type=float, default=1)

# for 2D-3D Contrastive CL and AE/VAE
parser.add_argument('--alpha_1', type=float, default=1)
parser.add_argument('--alpha_2', type=float, default=1)

# for 2D SSL and 3D-2D SSL
parser.add_argument('--SSL_2D_mode', type=str, default='AM')
parser.add_argument('--alpha_3', type=float, default=0.1)
parser.add_argument('--gamma_joao', type=float, default=0.1)
parser.add_argument('--gamma_joaov2', type=float, default=0.1)

# about if we would print out eval metric for training data
parser.add_argument('--eval_train', dest='eval_train', action='store_true')
parser.add_argument('--no_eval_train', dest='eval_train', action='store_false')
parser.set_defaults(eval_train=True)

parser.add_argument('--only_test', action='store_true')

parser.add_argument('--test_interval', type=int, default=1)

# about loading and saving
# parser.add_argument('--input_model_file', type=str, default='')
parser.add_argument('--output_model_dir', type=str, default='')

# verbosity
parser.add_argument('--verbose', dest='verbose', action='store_true')
parser.add_argument('--no_verbose', dest='verbose', action='store_false')
parser.set_defaults(verbose=False)



parser.add_argument('--zero_shot',action='store_true')
parser.add_argument('--prompt_id',nargs='+', type=int,default=None)
parser.add_argument('--prompt_policy',type=str,default='traversal', choices=['single','sample','traversal'])

parser.add_argument('--task_policy',type=str,default='multi_label', choices=['single','traversal','multi_mixture','multi_label'])

parser.add_argument('--model_name_or_path', type=str,)
parser.add_argument('--tokenizer_name',type=str)



parser.add_argument('--graph_transformer_graph_backbone',type=str,default='gin')
parser.add_argument('--graph_transformer_text_backbone',type=str,default='t5')
parser.add_argument('--attention_fasion',type=str,default='sequential')
parser.add_argument('--rich_features',action='store_true')
parser.add_argument('--transform_in_collator',action='store_true')

parser.add_argument('--single_split',type=int,default=None)

parser.add_argument('--output_result_to_file',type=str,default=None)

parser.add_argument('--few_shot',type=int,default=None)

parser.add_argument('--few_shot_fashion',type=str,default='finetune',choices=['finetune', 'linear','prompttune'])

parser.add_argument('--few_shot_prompt_fashion',type=str,default='traversal',choices=['traversal', 'max','max_abs'])

parser.add_argument('--not_retest_tasks_in_result_file',action='store_true')

parser.add_argument('--prompt_augmentation',default='',choices=['','rewrite','expande','detail','shorten','name'])
parser.add_argument('--prompt_augmentation_file_prefix',default='augmented')

parser.add_argument('--return_model_size',action='store_true')

parser.add_argument('--prompt_file',type=str,default='all_downstream_task_prompt_multitask.json')


args,left = parser.parse_known_args()
print('arguments\t', args)

assert args.graph_transformer_graph_backbone in ['graphormer', 'gin','kvplm','molt5','momu', 'galactica', 'gpt3']
if args.graph_transformer_graph_backbone == 'graphormer':
    parsernew = HfArgumentParser(GraphormerConfig)
    # parsernew = argparse.ArgumentParser()
    parsernew = GraphormerModel.add_args(parsernew)
    graph_args = parsernew.parse_args(left)
    graph_args=Graphormer_version_dict[graph_args.arch](graph_args)
    # print('graphormer_args',graphormer_args)
elif args.graph_transformer_graph_backbone == 'gin':
    parsernew = HfArgumentParser(GinConfig)
    graph_args = parsernew.parse_args(left)
elif args.graph_transformer_graph_backbone   == 'kvplm' or args.graph_transformer_text_backbone == 'kvplm':
    args.graph_transformer_graph_backbone = 'kvplm'
    args.graph_transformer_text_backbone = 'kvplm'
    parsernew = HfArgumentParser(KVPLMConfig)
    graph_args = parsernew.parse_args(left)
    args.tokenizer_name='allenai/scibert_scivocab_uncased'
elif args.graph_transformer_graph_backbone == 'momu' or args.graph_transformer_text_backbone == 'momu':
    args.graph_transformer_graph_backbone = 'momu'
    args.graph_transformer_text_backbone = 'momu'
    parsernew = HfArgumentParser(MoMuConfig)
    graph_args = parsernew.parse_args(left)
    args.tokenizer_name = 'allenai/scibert_scivocab_uncased'
elif args.graph_transformer_graph_backbone == 'molt5' or args.graph_transformer_text_backbone == 'molt5':
    args.graph_transformer_graph_backbone = 'molt5'
    args.graph_transformer_text_backbone = 'molt5'
    parsernew = HfArgumentParser(MolT5Config)
    graph_args = parsernew.parse_args(left)
    assert graph_args.init_checkpoint in ['laituan245/molt5-base','laituan245/molt5-small','laituan245/molt5-large']
    args.tokenizer_name = graph_args.init_checkpoint
elif args.graph_transformer_graph_backbone == 'galactica' or args.graph_transformer_text_backbone == 'galactica':
    args.graph_transformer_graph_backbone = 'galactica'
    args.graph_transformer_text_backbone = 'galactica'
    parsernew = HfArgumentParser(GalacticaConfig)
    graph_args = parsernew.parse_args(left)
    assert graph_args.init_checkpoint in ['facebook/galactica-1.3b','facebook/galactica-125m']
    args.tokenizer_name = graph_args.init_checkpoint
elif args.graph_transformer_graph_backbone == 'gpt3' or args.graph_transformer_text_backbone == 'gpt3':
    args.graph_transformer_graph_backbone = 'gpt3'
    args.graph_transformer_text_backbone = 'gpt3'
    parsernew = HfArgumentParser(GPT3Config)
    graph_args = parsernew.parse_args(left)
    assert graph_args.init_checkpoint in ['text-davinci-003']
    args.tokenizer_name = "mrsteyk/gpt3-tokenizer"
if args.few_shot is not None:
    assert args.few_shot % args.batch_size ==0 or args.few_shot<args.batch_size
    if args.few_shot<args.batch_size:
        print('Warning: the few shot number is smaller than batch size; setting batch_size to few_shot')
        args.grad_accum_step=int(args.grad_accum_step/(args.batch_size/args.few_shot))
        args.batch_size=args.few_shot
        print('batch_size: ',args.batch_size,'grad_accum_step: ',args.grad_accum_step)




assert args.batch_size % args.grad_accum_step==0
args.batch_size_ori=args.batch_size
args.batch_size=args.batch_size_ori//args.grad_accum_step



if args.graph_transformer_graph_backbone in ['kvplm','momu','galactica','gpt3']:
    if graph_args.init_checkpoint is None:
        graph_args.init_checkpoint = args.model_name_or_path
    if args.model_name_or_path is None:
        args.model_name_or_path = graph_args.init_checkpoint
 
        
        

if args.task_policy=='single':
    assert args.single_split is not None
else:
    if args.single_split is not None:
        print('single_split is specified as ',args.single_split,", but it will not be used in task_policy ",args.task_policy)

# Only when split and mixture all the labels, do split_label
args.split_label=args.task_policy=='multi'






def get_num_task(dataset):
    """ used in molecule_finetune.py """
    if dataset == 'tox21':
        return 12
    elif dataset in ['hiv', 'bace', 'bbbp', 'donor','esol','freesolv','lipo']:
        return 1
    elif dataset == 'pcba':
        return 128
    elif dataset == 'muv':
        return 17
    elif dataset == 'toxcast':
        return 617
    elif dataset == 'sider':
        return 27
    elif dataset == 'clintox':
        return 2
    elif dataset == 'cyp450':
        return 5
    raise ValueError(dataset + ': Invalid dataset name.')

def task_type(dataset):
    if dataset in ['esol','freesolv','lipo']:
        return 'reg'
    else:
        return 'cla'

def better_result(result,reference,dataset):
    if task_type(dataset)=='cla':
        return result>reference
    else:
        assert task_type(dataset)=='reg'
        return result<reference

def reg_thre_by_task(dataset):
    if dataset in ['esol']:
        return 10
    elif dataset == 'lipo':
        return 10
    elif dataset == 'freesolv':
        return 15
    else:
        raise ValueError('not supported task type for regression')



def train(model, device, loader, optimizer):
    model.train()
    total_loss = 0

    for step, batch in tqdm(enumerate(loader),disable=args.disable_tqdm):
        for key in batch.keys():
            batch[key] = batch[key].to(device)

        loss= model(**batch)['loss']/args.grad_accum_step
        loss.backward()
        if (step+1) % args.grad_accum_step == 0:
            optimizer.step()
            optimizer.zero_grad()
        total_loss += loss.detach().item()

    return total_loss / len(loader)*args.grad_accum_step

fig_number=0

def eval_result(model, device, loader,label_dict,tokenizer):
    if task_type(args.dataset)=='cla':
        model.eval()
        y_true, y_scores = [], []

        id_y=label_dict[1][0]
        id_n=label_dict[0][0]
        id_invalid=label_dict['invalid'][0]

        for step, batch in enumerate(loader):
            for key in batch.keys():
                batch[key] = batch[key].to(device)
            with torch.no_grad():
                labels=batch["labels"]
                del batch["labels"]
                if args.graph_transformer_graph_backbone in ['graphormer']: #Ours
                    batch["max_length"] = 3 # <PAD> CLASS <EOS>
                    output = model.generate(
                        **batch, output_scores=True, return_dict_in_generate=True
                        # num_beams=beam_size,
                        # no_repeat_ngram_size=no_repeat_ngram_size,
                    )
                    logits=output.scores[0].unsqueeze(1) #logits of CLASS
                elif args.graph_transformer_graph_backbone in ['galactica']:  # galactica
                    batch["max_new_tokens"] = 1 # <PAD> CLASS <EOS>
                    output = model.generate(
                        **batch, output_scores=True, return_dict_in_generate=True
                        # num_beams=beam_size,
                        # no_repeat_ngram_size=no_repeat_ngram_size,
                    )
                    logits=output.scores[0].unsqueeze(1) #logits of CLASS
                elif args.graph_transformer_graph_backbone in ['gpt3']:
                    prompt = tokenizer.batch_decode(batch["input_ids"])[0]  # llm only supports batch_size = 1
                    output = model.generate(prompt)
                    logits = output["choices"][0]["logprobs"]["top_logprobs"][0]
                else:
                    logits = model(**batch)['logits']
            index = labels != id_invalid #mask both text not answer and invalid labels; shape: [batch,answer length]
            if not isinstance(logits,dict): # for generative model

                assert logits[index].ndim==2 # selected answer shape:[n_valid_sample,n_vocabulary]

                pred=(logits[index][:, id_y] - logits[index][:, id_n]).view([-1,1])
                true = labels[index].view(pred.shape)
                true[true == id_y] = 1
                true[true == id_n] = 0
                true[true == id_invalid] = -100
            else: # for contrastive model, logits is dict
                if args.graph_transformer_graph_backbone in ['gpt3']:
                    positive_words = ["Yes", " Yes", "yes", " yes"]
                    negative_words = ["No", " No", "no", " no"]
                    positive_score = []
                    for word in positive_words:
                        if word in logits:
                            positive_score.append(logits[word])
                    positive_score = np.array(positive_score).mean()
                    negative_score = []
                    for word in negative_words:
                        if word in logits:
                            negative_score.append(logits[word])
                    negative_score = np.array(negative_score).mean()
                    pred = torch.tensor([positive_score - negative_score > 0]).unsqueeze(1)
                else:
                    pred = (logits['pos'].unsqueeze(1)[index] - logits['neg'].unsqueeze(1)[index]).view([-1, 1]) #shape of logits['pos] and logits['pos] are [batch]
                
                true = labels[index].view(pred.shape)
                true = labels[index].view(pred.shape)
                true[true == id_y] = 1
                true[true == id_n] = 0
                true[true == id_invalid] = -100
                assert torch.sum(true == id_invalid) == 0 # For contrastive model, invalid label is previously replaced by id_invalid(-100). Replace it here. Not necessary, because only valid label are selected


            y_true.append(true)
            y_scores.append(pred)

        y_true = torch.cat(y_true, dim=0).cpu().numpy()
        y_scores = torch.cat(y_scores, dim=0).cpu().numpy()

        roc_list = []
        for i in range(y_true.shape[1]):
            # AUC is only defined when there is at least one positive data.
            if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
                is_valid = y_true[:, i]  >= 0
                roc_list.append(roc_auc_score(y_true[is_valid, i], y_scores[is_valid, i]))
            else:
                print('{} is invalid'.format(i))

        if len(roc_list) < y_true.shape[1]:
            print(len(roc_list))
            print('Some target is missing!')
            print('Missing ratio: %f' % (1 - float(len(roc_list)) / y_true.shape[1]))

        if len(roc_list)==0:
            return {'score':0},0, y_true, y_scores
        else:
            return {'score':sum(roc_list) / len(roc_list)}, 0, y_true, y_scores
    else:
        model.eval()
        y_true, y_scores = [], []
        for step, batch in enumerate(loader):
            for key in batch.keys():
                batch[key] = batch[key].to(device)
            with torch.no_grad():
                labels=batch["labels"]
                del batch["labels"]
                if "decoder_attention_mask" in batch:
                    del batch["decoder_attention_mask"]
                if hasattr(model,'decoder'):
                    batch["max_length"] = labels.shape[1]
                    ids = model.generate(
                        **batch,
                        # num_beams=beam_size,
                        # no_repeat_ngram_size=no_repeat_ngram_size,
                    )
                else:
                    logits = model(**batch)['logits']

                    ids=logits.argmax(2)


            pred=[]
            for i in range(ids.shape[0]):
                ind_valid=labels[i, :]>=0
                if ind_valid.shape[0]>ids.shape[1]:
                    ind_valid=ind_valid[0:(ids.shape[1])]
                # pred.append(tokenizer.decode(logits[i,ind_valid,:].argmax(1)))
                pred.append(tokenizer.decode(ids[i, ind_valid]))
            pred_number=[]
            for result in pred:
                number_list=re.findall(r"-?\d+\.?\d*e??\d*?",result)
                try:
                    decoded_number=eval(number_list[0])
                except:
                    decoded_number=float(np.nan)

                pred_number.append(decoded_number)
            true=[]
            for i in range(labels.shape[0]):
                true.append(tokenizer.decode(labels[i, labels[i, :]>0]))
            true_number=[]
            for result in true:
                number_list=re.findall(r"-?\d+\.?\d*e??\d*?",result.replace(" ",""))
                true_number.append(eval((number_list[0])) if len(number_list)>0 else float(np.nan))


            y_true+=true_number
            y_scores+=pred_number

        y_true = torch.tensor(y_true)
        y_scores = torch.tensor(y_scores)

        ind=(~y_scores.isnan())&(y_scores.abs()<reg_thre_by_task(args.dataset))
        # ind = (~y_scores.isnan())
        ratio=ind.float().mean()
        y_true=y_true[ind]
        y_scores=y_scores[ind]

        mrs=(y_true-y_scores).std()
        naive_msr=(y_true-y_true.mean()).std()

        corrcoef=np.corrcoef(y_true,y_scores)[0,1]

        try:
            r2=r2_score(y_true,y_scores)
        except:
            r2=np.nan


        # plt.figure()
        # plt.scatter(y_true,y_scores)
        if args.plot_regression:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=y_true,
                y=y_scores,
                mode='markers',
                # name='GIN(sup)',
                marker=dict(
                    size=25,
                    opacity=0.5,
                    line=dict(width=2,
                              ), symbol="diamond"),
                # line=dict(width=8) if model_name(model) == 'Ours' else dict(width=6, dash='dashdot')
            ))
            fig.update_layout(
                title=args.dataset.replace('_',' '),
            )
            fig.update_layout(title={'font': {'size': 50}})

            fig.update_layout(
                xaxis_title='True Value',
                yaxis_title='Predicted Value',
                width=1000,
                height=1000,
                font=dict(
                    size=30,
                    color="Black"
                )
            )

            global fig_number
            fig.write_image('cache/{}fig{}.png'.format(args.dataset,fig_number))
            fig_number+=1

        # print(naive_msr)

        return {'ratio':float(ratio),'RMSE':float(mrs),'corrcoef':float(corrcoef),'R-Square':float(r2),'score':float(mrs)}, 0, y_true, y_scores


def downstream_task_by_transform(transform,model,train_loader,val_loader,test_loader,prompt=''):
    train_loader.dataset.transform=transform
    val_loader.dataset.transform = transform
    test_loader.dataset.transform = transform


    if args.few_shot:
        model = get_model(args, graph_args,tokenizer)
    
    model.to(device)
    if args.few_shot_fashion == 'finetune':
        model_param_group = [{'params': model.parameters()}]
    elif args.few_shot_fashion == 'linear':  #Only linear weight
        if args.graph_transformer_graph_backbone=='kvplm':
            model_param_group =[{'params': model.cls.predictions.decoder.parameters()}]
        elif args.graph_transformer_graph_backbone=='momu':
            model_param_group = [{'params': model.text_proj_head[2].parameters()}]
        else:
            model_param_group = [{'params': model.lm_head.parameters()}]
    elif args.few_shot_fashion == 'prompttune':
        model_param_group = [{'params': model.encoder.embed_tokens.parameters()}]
    else:
        raise ValueError("not supported few shot fashion")
    optimizer = optim.Adam(model_param_group, lr=args.lr,
                        weight_decay=args.decay)

    train_roc_list, val_roc_list, test_roc_list = [], [], []
    train_acc_list, val_acc_list, test_acc_list = [], [], []

    train_full_list, val_full_list, test_full_list = [], [], []

    best_val_roc, best_val_idx = None, 0



    if args.zero_shot:
        if args.eval_train and (not args.only_test):
            train_roc, train_acc, train_target, train_pred = eval_result(model, device, train_loader,label_dict,tokenizer)
        else:
            train_roc={'score':0}
        if not args.only_test:
            val_roc, val_acc, val_target, val_pred = eval_result(model, device, val_loader,label_dict,tokenizer)
        else:
            val_roc={'score':0}
        test_roc, test_acc, test_target, test_pred = eval_result(model, device, test_loader,label_dict,tokenizer)

        print(
            'train: {}\tval: {}\ttest: {}'.format(train_roc, val_roc,
                                                                   test_roc))
        model_file=args.model_name_or_path if args.model_name_or_path is not None else graph_args.init_checkpoint
        if not(test_roc['score']==0) and args.output_result_to_file is not None:
            print('Outputing result to '+args.output_result_to_file)
            record = [(args.dataset,test_loader.dataset.single_split, model_file, train_roc['score'], val_roc['score'], test_roc['score'], prompt)]
            df = pd.DataFrame(record,
                              columns=['dataset', 'split','model_name_or_path', 'train_roc', 'val_roc', 'test_roc', 'prompt'])
            #Saved csv have an extra first column of index, which is always 0 in this case.
            df.to_csv(args.output_result_to_file, mode='a', header=False)

    else:
        for epoch in range(1, args.epochs + 1):
            loss_acc = train(model, device, train_loader, optimizer)
            print('Epoch: {}\nLoss: {}'.format(epoch, loss_acc))

            if (epoch+1) % args.test_interval==0:

                if args.eval_train:
                    train_roc, train_acc, train_target, train_pred = eval_result(model, device, train_loader,label_dict,tokenizer)
                else:
                    train_acc = 0
                    train_roc = {'score':0}
                val_roc, val_acc, val_target, val_pred = eval_result(model, device, val_loader,label_dict,tokenizer)
                test_roc, test_acc, test_target, test_pred = eval_result(model, device, test_loader,label_dict,tokenizer)

                train_roc_list.append(train_roc['score'])
                train_acc_list.append(train_acc)
                train_full_list.append(train_roc)
                val_roc_list.append(val_roc['score'])
                val_acc_list.append(val_acc)
                val_full_list.append(val_roc)
                test_roc_list.append(test_roc['score'])
                test_acc_list.append(test_acc)
                test_full_list.append(test_roc)
                print(
                    'train: {}\tval: {}\ttest: {}'.format(train_roc, val_roc,
                                                          test_roc))
                print()

                if best_val_roc is None:
                    best_val_roc = val_roc['score']
                    assert best_val_idx == 0

                if better_result(val_roc['score'], best_val_roc,args.dataset):
                    best_val_roc = val_roc['score']
                    best_val_idx = epoch - 1
                    if not args.output_model_dir == '':
                        output_model_path = join(args.output_model_dir, 'model_best.pth')
                        # saved_model_dict = {
                        #     # 'molecule_model': molecule_model.state_dict(),
                        #     'model':
                        # }
                        torch.save(model.state_dict(), output_model_path)

                        filename = join(args.output_model_dir, 'evaluation_best.pth')
                        np.savez(filename, val_target=val_target, val_pred=val_pred,
                                 test_target=test_target, test_pred=test_pred)
        if max(val_roc_list) > 0:
            best_val_idx=val_roc_list.index(max(val_roc_list)) if task_type(args.dataset)=='cla' else val_roc_list.index(min(val_roc_list))
        else:
            best_val_idx=test_roc_list.index(max(test_roc_list)) if task_type(args.dataset)=='cla' else val_roc_list.index(min(test_roc_list))

        print(
            'best train: {}\tval: {}\ttest: {}'.format(train_full_list[best_val_idx], val_full_list[best_val_idx],
                                                                   test_full_list[best_val_idx]))

        model_file=args.model_name_or_path if args.model_name_or_path is not None else graph_args.init_checkpoint
        if not(test_roc_list[best_val_idx]==0) and args.output_result_to_file is not None:
            print('Outputing result to file '+args.output_result_to_file)
            record=[(args.dataset,test_loader.dataset.single_split,model_file,args.epochs,args.lr,args.runseed,best_val_idx,train_roc_list[best_val_idx],val_roc_list[best_val_idx],test_roc_list[best_val_idx],prompt)]
            df = pd.DataFrame(record,
                              columns=['dataset','split', 'model_name_or_path','epoch','lr','runseed','best_val_idx','train_best','valid_best','test_best','prompt'
                                       ])
            df.to_csv(args.output_result_to_file, mode='a', header=False)

        if args.output_model_dir is not '':
            output_model_path = join(args.output_model_dir, 'model_final.pth')
            torch.save(model.state_dict(), output_model_path)



if __name__ == '__main__':
    torch.manual_seed(args.runseed)
    np.random.seed(args.runseed)
    device = torch.device('cuda:' + str(args.device)) \
        if (torch.cuda.is_available() and not args.no_cuda) else torch.device('cpu')
    if torch.cuda.is_available() and not args.no_cuda :
        torch.cuda.manual_seed_all(args.runseed)


    tokenizer_kwargs = {
        "cache_dir": None,
        "use_fast": True,
        "revision": 'main',
        "use_auth_token": None,
    }
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, **tokenizer_kwargs)

    if args.few_shot and args.few_shot_prompt_fashion!='traversal':

        def modify_name(name):
            name = name.replace('.ckpt', '.pt')
            name=name.replace('ckpts/','')
            if name[-1]=='/':
                name=name[:-1]
            return name

        file_name=os.path.join('cache','result_'+args.few_shot_prompt_fashion+'_prompt_table.csv')
        prompts_pd = pd.read_csv(file_name,index_col='unique_task_id')
        rename_keys={}
        for name in prompts_pd.columns:
            rename_keys[name]=modify_name(name)
        prompts_pd=prompts_pd.rename(columns=rename_keys)
        prompt={}
        model_name=modify_name(args.model_name_or_path)
        for ind in range(get_num_task(args.dataset)):
            if args.dataset + '@' + str(ind) in prompts_pd.index.values:
                res=prompts_pd.loc[args.dataset+'@'+str(ind),model_name]
                if pd.isna(res):
                    continue
                prompt[str(ind)]=[res]

    else:
        if args.prompt_augmentation=='':
            with open(os.path.join("prompts",args.prompt_file), 'r') as load_f:
                prompts = commentjson.load(load_f)
            prompt=prompts[args.dataset]
        else:
            with open(os.path.join("prompts",args.prompt_file), 'r') as load_f:
                prompts = commentjson.load(load_f)
            prompt_all=prompts[args.dataset]
            prompt={}
            for key in prompt_all:
                prompt[key]=prompt_all[key][args.prompt_augmentation]


    if isinstance(prompt,list):
        prompt_token=tokenizer(prompt,return_special_tokens_mask=True)
        input_ids = [item for item in prompt_token.data['input_ids']]
        attention_mask = [item for item in prompt_token.data['attention_mask']]
        if args.prompt_id is None:
            args.prompt_id = list(range(len(prompt)))
    elif isinstance(prompt,dict):
        prompt_token={}
        input_ids={}
        attention_mask={}
        args.prompt_id={}
        for key in prompt.keys():
            if len(prompt[key])>0:
                prompt_token[key]=tokenizer(prompt[key],return_special_tokens_mask=True)
                input_ids[key] = [item for item in prompt_token[key].data['input_ids']]
                attention_mask[key] = [item for item in prompt_token[key].data['attention_mask']]
                args.prompt_id[key] = list(range(len(prompt[key])))
    else:
        raise ValueError('Prompt type not supported. Only list or dict of (list of) prompts are supported.')


    label_ignore = [-100]
    raw_label = {1: 'Yes', 0: 'No', 'invalid': label_ignore}
    label_y = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(raw_label[1])) # Not include CLS or other tokens
    label_n = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(raw_label[0]))
     # input a list so that they can be concatenated in collator
    label_dict = {1: label_y, 0: label_n, 'invalid': label_ignore}


    # Bunch of classification tasks
    num_tasks = get_num_task(args.dataset)
    dataset_folder = 'property_data/'



    if args.graph_transformer_graph_backbone in ['kvplm', 'galactica','gpt3']:
        dataset = MoleculeDatasetSplitLabel(root=dataset_folder, name=args.dataset,return_smiles=True,split_label=args.split_label,single_split=args.single_split)
    else:
        dataset = MoleculeDatasetSplitLabel(root=dataset_folder, name=args.dataset,split_label=args.split_label,single_split=args.single_split)

    print(dataset)
    print(dataset[0])



    if args.split == 'scaffold':
        # if args.single_split is not None:
        smiles_list = pd.read_csv(dataset_folder + args.dataset + '/processed/smiles.csv',
                                  header=None)[0].tolist()
        train_index, valid_index, test_index = scaffold_split(
            torch.arange(len(smiles_list)), smiles_list, null_value=0, frac_train=0.8,
            frac_valid=0.1, frac_test=0.1)

        train_index_total=[]
        valid_index_total=[]
        test_index_total=[]
        for times in range(dataset.label_number):
            train_index_times=train_index+times*dataset.len_oridata()
            valid_index_times = valid_index + times * dataset.len_oridata()
            test_index_times = test_index + times * dataset.len_oridata()

            train_index_total.append(train_index_times)
            valid_index_total.append(valid_index_times)
            test_index_total.append(test_index_times)
        train_index_total=torch.cat(train_index_total,0)
        valid_index_total=torch.cat(valid_index_total,0)
        test_index_total=torch.cat(test_index_total,0)



        train_dataset = dataset[train_index_total]
        valid_dataset = dataset[valid_index_total]
        test_dataset = dataset[test_index_total]



        print('split via scaffold')
    elif args.split == 'random':
        train_dataset, valid_dataset, test_dataset = random_split(
            dataset, null_value=0, frac_train=0.8, frac_valid=0.1,
            frac_test=0.1, seed=args.seed)
        print('randomly split')
    elif args.split == 'random_scaffold':
        smiles_list = pd.read_csv(dataset_folder + args.dataset + '/processed/smiles.csv',
                                  header=None)[0].tolist()
        train_dataset, valid_dataset, test_dataset = random_scaffold_split(
            dataset, smiles_list, null_value=0, frac_train=0.8,
            frac_valid=0.1, frac_test=0.1, seed=args.seed)
        print('random scaffold')
    else:
        raise ValueError('Invalid split option.')
    print(train_dataset[0])


    data_collator = GraphTransformer_collator_dict[args.graph_transformer_graph_backbone][
        args.graph_transformer_text_backbone](
        tokenizer=tokenizer,
        transform_in_collator=args.transform_in_collator,
        rich_features=args.rich_features
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers,collate_fn=data_collator)
    val_loader = DataLoader(valid_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers,collate_fn=data_collator)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers,collate_fn=data_collator)

    model=get_model(args,graph_args,tokenizer)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    if args.return_model_size:
        print('Model size: {}'.format(count_parameters(model)))


    if args.task_policy =='traversal':
        recurrent_range=range(num_tasks)
    elif args.task_policy =='single':
        recurrent_range = [args.single_split]
    else:
        recurrent_range = range(1)

    if args.zero_shot:
        columns = ['dataset', 'split','model_name_or_path', 'train_roc', 'val_roc', 'test_roc', 'prompt']
    else:
        columns=['dataset','split', 'model_name_or_path','epoch','lr','runseed','best_val_idx','train_best','valid_best','test_best','prompt']
    if args.not_retest_tasks_in_result_file:
        if os.path.exists(args.output_result_to_file):
            result_file=pd.read_csv(args.output_result_to_file,header=None,index_col=0)
            result_file.columns=columns
        else:
            result_file=pd.DataFrame(columns=columns)

    for single_split_label in recurrent_range:
        if args.task_policy in ['traversal','single']:
            print('label split: ',single_split_label)
        if not str(single_split_label) in prompt:
            print('No prompt for label split {}'.format(single_split_label))
            continue
        if args.not_retest_tasks_in_result_file:
            if len(result_file[(result_file['dataset']==args.dataset) & (result_file['split']==single_split_label)])>0:
                print(args.dataset,' ',single_split_label,'has been tested')
                continue

        train_loader.dataset.set_single_split(single_split_label)
        val_loader.dataset.set_single_split(single_split_label)
        test_loader.dataset.set_single_split(single_split_label)

        dataset.set_single_split(single_split_label)
        if args.few_shot is not None:
            ind_each_class = {}
            for ind in train_index_total:
                label=int(dataset[ind].y)
                if label not in ind_each_class:
                    ind_each_class[label]=[ind]
                else:
                    ind_each_class[label].append(ind)

            for key in ind_each_class.keys():
                ind_each_class[key]=np.random.choice(ind_each_class[key], size=min(len(ind_each_class[key]),args.few_shot),replace=False).tolist()
            train_index_total=[]
            for key in ind_each_class.keys():
                train_index_total+=ind_each_class[key]

        train_dataset = dataset[train_index_total]
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=args.num_workers, collate_fn=data_collator)
        train_loader.dataset.set_single_split(single_split_label)


        if args.prompt_policy == 'single':
            print(prompt[args.prompt_id[0]])
            downstream_task_by_transform(
                lambda x: add_prompt_conditional_generation_transform_single_dict[args.graph_transformer_graph_backbone](
                    data=x,data_label=x.y,input_ids=input_ids[args.prompt_id[0]],
                    attention_mask=attention_mask[args.prompt_id[0]],label_dict=label_dict,
                    rich_features=args.rich_features,transform_in_collator=args.transform_in_collator,raw_prompts=prompt[args.prompt_id[0]],raw_label=raw_label,tokenizer=tokenizer,generaltive_label=(task_type(args.dataset)=='reg')),
                model,train_loader,val_loader,test_loader,prompt[args.prompt_id[0]])
        elif args.prompt_policy == 'sample':
            downstream_task_by_transform(
                lambda x: add_prompt_conditional_generation_transform_sample_dict[args.graph_transformer_graph_backbone]
                (data=x,data_label=x.y,input_ids=input_ids,
                    attention_mask=attention_mask,label_dict=label_dict,
                    prompt_ids=args.prompt_id,rich_features=args.rich_features,transform_in_collator=args.transform_in_collator,raw_prompts=prompt,raw_label=raw_label,tokenizer=tokenizer,generaltive_label=(task_type(args.dataset)=='reg')),
                model,train_loader,val_loader,test_loader)
        elif args.prompt_policy == 'traversal':
            for prompt_id in args.prompt_id[str(single_split_label)]:
                print(prompt[str(single_split_label)][prompt_id])
                # with open('result_graph_transformer.log', 'a+') as f:
                #     f.write(prompt[prompt_id])
                #     f.write('\n')
                downstream_task_by_transform(
                    lambda x: add_prompt_conditional_generation_transform_single_dict[args.graph_transformer_graph_backbone]
                    (data=x,data_label=x.y,input_ids=input_ids[str(single_split_label)][prompt_id],
                    attention_mask=attention_mask[str(single_split_label)][prompt_id],label_dict=label_dict,
                    rich_features=args.rich_features,transform_in_collator=args.transform_in_collator,raw_prompts=prompt[str(single_split_label)][prompt_id],raw_label=raw_label,tokenizer=tokenizer,generaltive_label=(task_type(args.dataset)=='reg')),
                    model,train_loader,val_loader,test_loader,prompt[str(single_split_label)][prompt_id])






