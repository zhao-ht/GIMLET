import torch
import numpy as np
# from . import algos

import pyximport
from ogb.utils import smiles2graph

pyximport.install(setup_args={"include_dirs": np.get_include()})
from .graphormer_transform import graphormer_data_transform
import numbers
# @torch.jit.script
def convert_to_single_emb(x, offset: int = 512):
    feature_num = x.shape[1] if len(x.shape) > 1 else 1
    feature_offset = (1 + np.arange(0, feature_num * offset, offset)).astype("long")
    x = x + feature_offset
    return x



def gin_add_prompt_conditional_generation_transform_single(data,data_label,input_ids,attention_mask,label_dict,**kwargs):
    if float(data_label) in label_dict:
        label=label_dict[float(data_label)]
    else:
        label=label_dict['invalid']

    input_ids_out=input_ids
    attention_mask_out=attention_mask
    return {'graph': data,
            'input_ids': input_ids_out,
            'attention_mask': attention_mask_out,
            # 'special_tokens_mask': special_tokens_mask,
            'labels': label,
            }


def gimlet_add_prompt_conditional_generation_transform_single(data, data_label, input_ids, attention_mask, label_dict, transform_in_collator, rich_features=False, tokenizer=None, generaltive_label=False, **kwargs):
    if generaltive_label:
        if isinstance(data_label,torch.Tensor) or isinstance(data_label, numbers.Number):
            label=tokenizer(str(round(float(data_label), 2)))
        else:
            label = tokenizer(str(data_label))
    else:
        if float(data_label) in label_dict:
            label=label_dict[float(data_label)]
        else:
            label=label_dict['invalid']

    input_ids_out=input_ids
    attention_mask_out=attention_mask

    if not transform_in_collator:
        data= graphormer_data_transform(data, rich_features=rich_features)
    if generaltive_label:
        return {'graph': data,
            'input_ids': input_ids_out,
            'attention_mask': attention_mask_out,
            # 'special_tokens_mask': special_tokens_mask,
            'labels': label['input_ids'],
                'decoder_attention_mask':label['attention_mask']
            }
    else:
        return {'graph': data,
                'input_ids': input_ids_out,
                'attention_mask': attention_mask_out,
                # 'special_tokens_mask': special_tokens_mask,
                'labels': label,
                }


def tokenize_function_gin_T5(examples,tokenizer,text_column_name,padding,max_seq_length,rich_features,transform_in_collator,**kwargs):
    # Remove empty lines
    # examples[text_column_name] = [line for line in examples[text_column_name] if len(line) > 0 and not line.isspace()]
    text = tokenizer(
        examples[text_column_name] if isinstance(examples[text_column_name],str) else examples[text_column_name][0],
        padding=padding,
        truncation=True,
        max_length=max_seq_length,
        # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
        # receives the `special_tokens_mask`.
        return_special_tokens_mask=True,
    )
    labels = tokenizer(
        examples['label'],
        padding=padding,
        truncation=True,
        max_length=max_seq_length,
        # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
        # receives the `special_tokens_mask`.
        return_special_tokens_mask=True,
    )


    # graph_data = smiles2graph(examples['graph'])
    if not transform_in_collator:
        graph_data = smiles2graph(examples['graph'])
    else:
        graph_data = examples['graph']

    return {'graph': graph_data,
            'input_ids': text.data['input_ids'],
            'attention_mask': text.data['attention_mask'],
            'special_tokens_mask': text.data['special_tokens_mask'],
            'labels': labels.data['input_ids']}


def tokenize_function_gimlet(examples, tokenizer, text_column_name, padding, max_seq_length, rich_features, transform_in_collator):
    # Remove empty lines
    # examples[text_column_name] = [line for line in examples[text_column_name] if len(line) > 0 and not line.isspace()]
    text = tokenizer(
        examples[text_column_name] if isinstance(examples[text_column_name],str) else examples[text_column_name][0], # if examples[text_column_name] is list
        padding=padding,
        truncation=True,
        max_length=max_seq_length,
        # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
        # receives the `special_tokens_mask`.
        return_special_tokens_mask=True,
    )
    labels = tokenizer(
        str(examples['label']),
        padding=padding,
        truncation=True,
        max_length=max_seq_length,
        # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
        # receives the `special_tokens_mask`.
        return_special_tokens_mask=True,
    )


    graph_data = examples['graph']


    if not transform_in_collator:
        graph_data = smiles2graph(examples['graph'])
        graph_data = graphormer_data_transform(graph_data,rich_features)

    return {'graph': graph_data,
            'input_ids': text.data['input_ids'],
            'attention_mask': text.data['attention_mask'],
            'special_tokens_mask': text.data['special_tokens_mask'],
            'labels': labels.data['input_ids']}