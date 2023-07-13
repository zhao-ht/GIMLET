from model import GIMLETConfig,GinConfig,KVPLMConfig,MoMuConfig,MolT5Config, GalacticaConfig, GPT3Config

import numpy as np
import torch
from sklearn.metrics import (r2_score,
                             roc_auc_score)
import plotly.graph_objects as go
from transformers import (
    HfArgumentParser,
)
import re


def load_graph_args(args,left):
    assert args.transformer_backbone in ['gimlet', 'gint5','kvplm','molt5','momu', 'galactica', 'gpt3']
    if args.transformer_backbone == 'gimlet':
        parsernew = HfArgumentParser(GIMLETConfig)
        graph_args = parsernew.parse_args(left)
    elif args.transformer_backbone == 'gint5':
        parsernew = HfArgumentParser(GinConfig)
        graph_args = parsernew.parse_args(left)
    elif args.transformer_backbone   == 'kvplm' :
        parsernew = HfArgumentParser(KVPLMConfig)
        graph_args = parsernew.parse_args(left)
        args.tokenizer_name='allenai/scibert_scivocab_uncased'
    elif args.transformer_backbone == 'momu' :
        parsernew = HfArgumentParser(MoMuConfig)
        graph_args = parsernew.parse_args(left)
        args.tokenizer_name = 'allenai/scibert_scivocab_uncased'
    elif args.transformer_backbone == 'molt5' :
        parsernew = HfArgumentParser(MolT5Config)
        graph_args = parsernew.parse_args(left)
        assert graph_args.init_checkpoint in ['laituan245/molt5-base','laituan245/molt5-small','laituan245/molt5-large']
        args.tokenizer_name = graph_args.init_checkpoint
    elif args.transformer_backbone == 'galactica' :
        parsernew = HfArgumentParser(GalacticaConfig)
        graph_args = parsernew.parse_args(left)
        assert graph_args.init_checkpoint in ['facebook/galactica-1.3b','facebook/galactica-125m']
        args.tokenizer_name = graph_args.init_checkpoint
    elif args.transformer_backbone == 'gpt3' :
        parsernew = HfArgumentParser(GPT3Config)
        graph_args = parsernew.parse_args(left)
        assert graph_args.init_checkpoint in ['text-davinci-003']
        args.tokenizer_name = "mrsteyk/gpt3-tokenizer"
    if args.transformer_backbone in ['kvplm','momu','galactica','gpt3']:
        if graph_args.init_checkpoint is None:
            graph_args.init_checkpoint = args.model_name_or_path
        if args.model_name_or_path is None:
            args.model_name_or_path = graph_args.init_checkpoint

    return args,graph_args



def eval_result(model, loader,label_dict,tokenizer,task_type,transformer_backbone,args=None):
    if task_type=='cla':
        model.eval()
        y_true, y_scores = [], []

        id_y=label_dict[1][0]
        id_n=label_dict[0][0]
        id_invalid=label_dict['invalid'][0]

        for step, batch in enumerate(loader):
            for key in batch.keys():
                batch[key] = batch[key].to(model.device)
            with torch.no_grad():
                labels=batch["labels"]
                if labels.shape[1]>1 and not transformer_backbone in ['kvplm']: # Yes <s>
                    assert all((labels[:,1]==tokenizer.eos_token_id) + (labels[:,1]==id_invalid))
                    labels=labels[:,0].unsqueeze(1)
                del batch["labels"]

                if transformer_backbone in ['gimlet']: #Ours
                    batch["max_length"] = 3 # <PAD> CLASS <EOS>
                    output = model.generate(
                        **batch, output_scores=True, return_dict_in_generate=True
                        # num_beams=beam_size,
                        # no_repeat_ngram_size=no_repeat_ngram_size,
                    )
                    logits=output.scores[0].unsqueeze(1) #logits of CLASS

                elif transformer_backbone in ['galactica']:  # galactica
                    batch["max_new_tokens"] = 1 # <PAD> CLASS <EOS>
                    output = model.generate(
                        **batch, output_scores=True, return_dict_in_generate=True
                        # num_beams=beam_size,
                        # no_repeat_ngram_size=no_repeat_ngram_size,
                    )
                    logits=output.scores[0].unsqueeze(1) #logits of CLASS

                elif transformer_backbone in ['gpt3']:
                    prompt = tokenizer.batch_decode(batch["input_ids"])[0]  # llm only supports batch_size = 1
                    output = model.generate(prompt)
                    logits = output["choices"][0]["logprobs"]["top_logprobs"][0]

                else: #kvplm and momu
                    logits = model(**batch)['logits']

            index = labels != id_invalid #mask both text not answer and invalid labels; shape: [batch,answer length]

            if not isinstance(logits,dict): # for generative model
                assert logits[index].ndim==2 # selected answer shape:[n_valid_sample,n_vocabulary]

                pred=(logits[index][:, id_y] - logits[index][:, id_n]).view([-1,1])
                true = labels[index].view(pred.shape)
                true[true == id_y] = 1
                true[true == id_n] = 0
                true[true == id_invalid] = -100

            else: # for contrastive model and gpt, logits is dict

                if transformer_backbone in ['gpt3']:
                    positive_words = ["Yes"]
                    negative_words = ["No"]
                    positive_score = []
                    for word in positive_words:
                        if word in logits:
                            positive_score.append(logits[word])
                    positive_score = np.array(positive_score).max()
                    negative_score = []
                    for word in negative_words:
                        if word in logits:
                            negative_score.append(logits[word])
                    negative_score = np.array(negative_score).max()
                    pred = torch.tensor([positive_score - negative_score > 0]).unsqueeze(1)

                else: #Momu
                    pred = (logits['pos'].unsqueeze(1)[index] - logits['neg'].unsqueeze(1)[index]).view([-1, 1]) #shape of logits['pos] and logits['pos] are [batch]

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

    else: # for regression

        model.eval()
        y_true, y_scores = [], []

        for step, batch in enumerate(loader):
            for key in batch.keys():
                batch[key] = batch[key].to(model.device)
            with torch.no_grad():
                labels=batch["labels"]
                del batch["labels"]
                if "decoder_attention_mask" in batch:
                    del batch["decoder_attention_mask"]

                if transformer_backbone in ['gimlet']: #Ours
                    batch["max_length"] = labels.shape[1]+1 # additional <pad> in the begining
                    ids = model.generate(
                        **batch,
                        # num_beams=beam_size,
                        # no_repeat_ngram_size=no_repeat_ngram_size,
                    )
                    pred = []
                    for i in range(ids.shape[0]):
                        pred.append(tokenizer.decode(ids[i, :]))

                elif transformer_backbone in ['galactica']:  # galactica
                    batch["max_new_tokens"] = labels.shape[1]+1 # <PAD> CLASS <EOS>
                    ids = model.generate(
                        **batch
                        # num_beams=beam_size,
                        # no_repeat_ngram_size=no_repeat_ngram_size,
                    )
                    ids=ids[:,batch['input_ids'].shape[1]:]
                    pred = []
                    for i in range(ids.shape[0]):
                        pred.append(tokenizer.decode(ids[i, :]))

                else: #kvplm
                    logits = model(**batch)['logits']
                    ids=logits.argmax(2)
                    pred = []
                    for i in range(ids.shape[0]):
                        ind_valid = labels[i, :] >= 0
                        if ind_valid.shape[0] > ids.shape[1]:
                            ind_valid = ind_valid[0:(ids.shape[1])]
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

        ind = (~y_scores.isnan())
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

        if args.plot_regression:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=y_true,
                y=y_scores,
                mode='markers',
                marker=dict(
                    size=25,
                    opacity=0.5,
                    line=dict(width=2,
                              ), symbol="diamond"),
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
            fig.write_image('cache/'+('{}_{}_fig{}.png'.format(args.dataset,args.model_name_or_path,fig_number)).replace('/','_'))
            fig_number+=1

        return {'ratio':float(ratio),'RMSE':float(mrs),'corrcoef':float(corrcoef),'R-Square':float(r2),'score':float(mrs)}, 0, y_true, y_scores
