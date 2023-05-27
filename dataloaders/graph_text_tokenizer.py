from ogb.utils import smiles2graph
from .graph_text_transform import graphormer_data_transform



def tokenize_function_gin_T5(examples,tokenizer,text_column_name,padding,max_seq_length,rich_features,transform_in_collator,**kwargs):
    # Remove empty lines
    # examples[text_column_name] = [line for line in examples[text_column_name] if len(line) > 0 and not line.isspace()]
    text = tokenizer(
        examples[text_column_name],
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

    # 在这里转换tensor是没有意义的，需要在collater里面转换，因为dataset会被缓存后再加载，缓存会把tensor变成list
    # graph_data = smiles2graph(examples['graph'])
    if not transform_in_collator:
        graph_data = smiles2graph(examples['graph'])
    else:
        graph_data = examples['graph']

    # graph_data={'x':torch.tensor(graph_data['node_feat']).long(),
    #             'edge_index':torch.tensor(graph_data['edge_index']).long(),
    #             'edge_attr':torch.tensor(graph_data['edge_feat']).long()}
    return {'graph': graph_data,
            'input_ids': text.data['input_ids'],
            'attention_mask': text.data['attention_mask'],
            'special_tokens_mask': text.data['special_tokens_mask'],
            'labels': labels.data['input_ids']}

def tokenize_function_graphormer_T5(examples,tokenizer,text_column_name,padding,max_seq_length,rich_features,transform_in_collator):
    # Remove empty lines
    # examples[text_column_name] = [line for line in examples[text_column_name] if len(line) > 0 and not line.isspace()]
    text = tokenizer(
        examples[text_column_name],
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

    # 在这里转换tensor是没有意义的，需要在collater里面转换，因为dataset会被缓存后再加载，缓存会把tensor变成list
    # graph_data = smiles2graph(examples['graph'])
    graph_data = examples['graph']
    # print(graph_data['node_feat'])
    # print(graph_data['edge_feat'])
    # if not rich_features:
    #     graph_data['node_feat'] = graph_data['node_feat'][:,0:2]
    #     graph_data['edge_feat'] = graph_data['edge_feat'][:,0:2]

    if not transform_in_collator:
        graph_data = smiles2graph(examples['graph'])
        graph_data = graphormer_data_transform(graph_data,rich_features)
    # graph_data={'x':torch.tensor(graph_data['node_feat']).long(),
    #             'edge_index':torch.tensor(graph_data['edge_index']).long(),
    #             'edge_attr':torch.tensor(graph_data['edge_feat']).long()}
    return {'graph': graph_data,
            'input_ids': text.data['input_ids'],
            'attention_mask': text.data['attention_mask'],
            'special_tokens_mask': text.data['special_tokens_mask'],
            'labels': labels.data['input_ids']}



def graphormer_transform_for_dataset(examples,rich_features):
    graph_data = smiles2graph(examples['graph'])
    graph_data = graphormer_data_transform(graph_data, rich_features)
    examples['graph']=graph_data
    return examples

def tokenize_function_graphormer_multitask(examples,rich_features,transform_in_collator):
    # Remove empty lines
    # examples[text_column_name] = [line for line in examples[text_column_name] if len(line) > 0 and not line.isspace()]
    # text = tokenizer(
    #     examples[text_column_name],
    #     padding=padding,
    #     truncation=True,
    #     max_length=max_seq_length,
    #     # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
    #     # receives the `special_tokens_mask`.
    #     return_special_tokens_mask=True,
    # )
    # labels = tokenizer(
    #     examples['label'],
    #     padding=padding,
    #     truncation=True,
    #     max_length=max_seq_length,
    #     # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
    #     # receives the `special_tokens_mask`.
    #     return_special_tokens_mask=True,
    # )

    # 在这里转换tensor是没有意义的，需要在collater里面转换，因为dataset会被缓存后再加载，缓存会把tensor变成list
    graph_data = smiles2graph(examples['graph'])

    # from pretrain_datasets.ChEMBL_STRING.mole_key import key_int, key_float

    label_cla=[]
    label_reg=[]
    for key in examples.keys():
        if len(key)>=4 and key[0:4]=='cla_':
            label_cla.append(examples[key])
        elif len(key)>=4 and key[0:4]=='reg_':
            label_reg.append(examples[key])

    # print(graph_data['node_feat'])
    # print(graph_data['edge_feat'])
    # if not rich_features:
    #     graph_data['node_feat'] = graph_data['node_feat'][:,0:2]
    #     graph_data['edge_feat'] = graph_data['edge_feat'][:,0:2]

    if not transform_in_collator:
        graph_data = graphormer_data_transform(graph_data,rich_features)
    # graph_data={'x':torch.tensor(graph_data['node_feat']).long(),
    #             'edge_index':torch.tensor(graph_data['edge_index']).long(),
    #             'edge_attr':torch.tensor(graph_data['edge_feat']).long()}
    return {'graph': graph_data,
            'label_cla': label_cla,
            'label_reg': label_reg,}