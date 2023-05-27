from torch_geometric.data.dataloader import Collater
import torch
from typing import Optional

from transformers import (
    DataCollatorForLanguageModeling,
)
from transformers.tokenization_utils_base import BatchEncoding

from collections.abc import Mapping, Sequence

import torch.utils.data
from torch.utils.data.dataloader import default_collate

from torch_geometric.data import Data, Batch
import numpy as np
from .graph_text_transform import graphormer_data_transform_tensor
from ogb.utils import smiles2graph
import time

def pad_1d_unsqueeze(x, padlen):
    x = x + 1  # pad id = 0
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen], dtype=x.dtype)
        new_x[:xlen] = x
        x = new_x
    return x.unsqueeze(0)


def pad_2d_unsqueeze(x, padlen):
    x = x + 1  # pad id = 0
    xlen, xdim = x.size()
    if xlen < padlen:
        new_x = x.new_zeros([padlen, xdim], dtype=x.dtype)
        new_x[:xlen, :] = x
        x = new_x
    return x.unsqueeze(0)


def pad_attn_bias_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype).fill_(float("-inf"))
        new_x[:xlen, :xlen] = x
        new_x[xlen:, :xlen] = 0
        x = new_x
    return x.unsqueeze(0)


def pad_edge_type_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen, x.size(-1)], dtype=x.dtype)
        new_x[:xlen, :xlen, :] = x
        x = new_x
    return x.unsqueeze(0)


def pad_spatial_pos_unsqueeze(x, padlen):
    x = x + 1
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype)
        new_x[:xlen, :xlen] = x
        x = new_x
    return x.unsqueeze(0)


def pad_3d_unsqueeze(x, padlen1, padlen2, padlen3):

    x = x + 1
    xlen1, xlen2, xlen3, xlen4 = x.size()
    if xlen1 < padlen1 or xlen2 < padlen2 or xlen3 < padlen3:
        # new_x = x.new_zeros([padlen1, padlen2, padlen3, xlen4], dtype=x.dtype)
        # new_x[:xlen1, :xlen2, :xlen3, :] = x
        # x = new_x
        x = torch.nn.functional.pad(
            x,
            pad=(0,0, 0, padlen3 - xlen3,0, padlen2 - xlen2,0, padlen1 - xlen1, ),
            mode='constant',
            value=0
        )

    return x.unsqueeze(0)

def pad_3d_sequences(xs, padlen1, padlen2, padlen3):
    result=torch.zeros([len(xs),padlen1,padlen2,padlen3,xs[0].shape[-1]], dtype=xs[0].dtype)
    for i,x in enumerate(xs):
        result[i,:x.shape[0], :x.shape[1], :x.shape[2], :]=x+1
    return result


def pad(encoded_inputs,
        pad_token_id,
        pad_token_type_id,
        max_length: Optional = None,
        pad_to_multiple_of: Optional = None,
        return_attention_mask: Optional = None,
        padding_side='right',
        ):
    if return_attention_mask is None:
        return_attention_mask = "attention_mask" in encoded_inputs

    required_input = encoded_inputs[list(encoded_inputs.keys())[0]]


    if max_length is not None and pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
        max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of

    needs_to_be_padded =  len(required_input) != max_length

    # Initialize attention mask if not present.
    if return_attention_mask and "attention_mask" not in encoded_inputs:
        encoded_inputs["attention_mask"] = [1] * len(required_input)

    if needs_to_be_padded:
        difference = max_length - len(required_input)

        if padding_side == "right":
            if return_attention_mask:
                encoded_inputs["attention_mask"] = encoded_inputs["attention_mask"] + [0] * difference
            if "token_type_ids" in encoded_inputs:
                encoded_inputs["token_type_ids"] = (
                        encoded_inputs["token_type_ids"] + [pad_token_type_id] * difference
                )
            if "special_tokens_mask" in encoded_inputs:
                encoded_inputs["special_tokens_mask"] = encoded_inputs["special_tokens_mask"] + [1] * difference
            if "answer_mask" in encoded_inputs:
                encoded_inputs["answer_mask"] = encoded_inputs["answer_mask"] + [0] * difference
            if "input_ids" in encoded_inputs:
                encoded_inputs['input_ids'] = encoded_inputs['input_ids'] + [pad_token_id] * difference
            if "labels" in encoded_inputs:
                encoded_inputs['labels'] = encoded_inputs['labels'] + [-100] * difference #-100 can makesure that generation loss will not consider padded position
            if 'decoder_attention_mask' in encoded_inputs:
                encoded_inputs['decoder_attention_mask'] = encoded_inputs['decoder_attention_mask']+[0]*difference


        else:
            raise ValueError("Invalid padding strategy:" + str(padding_side))

    return encoded_inputs

def padding(encoded_inputs,
        pad_token_id,
        pad_token_type_id,
        max_length: Optional = None,
        pad_to_multiple_of: Optional = None,
        return_attention_mask: Optional = None):
    if isinstance(encoded_inputs, (list, tuple)) and isinstance(encoded_inputs[0], Mapping):
        encoded_inputs = {key: [example[key] for example in encoded_inputs] for key in encoded_inputs[0].keys()}


    required_input = encoded_inputs[list(encoded_inputs.keys())[0]]
    if required_input and not isinstance(required_input[0], (list, tuple)):
        encoded_inputs = pad(
            encoded_inputs,
            pad_token_id,
            pad_token_type_id,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            return_attention_mask=return_attention_mask,
        )
        return BatchEncoding(encoded_inputs, tensor_type='pt')

    batch_size = len(required_input)
    max_length = max(len(inputs) for inputs in required_input)
    # padding_strategy = PaddingStrategy.MAX_LENGTH


    batch_outputs = {}
    for i in range(batch_size):
        inputs = dict((k, v[i]) for k, v in encoded_inputs.items())
        outputs = pad(
            inputs,
            pad_token_id,
            pad_token_type_id,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            return_attention_mask=return_attention_mask,
        )

        for key, value in outputs.items():
            if key not in batch_outputs:
                batch_outputs[key] = []
            batch_outputs[key].append(value)
    return BatchEncoding(batch_outputs, tensor_type='pt')


def collator_graph_data(items, max_node=512, multi_hop_max_dist=20, spatial_pos_max=20,transform_in_collator=False,include_y=False,rich_features=False):
    items_new=[]
    ys=[]

    for item in items:

        if transform_in_collator:
            if isinstance(item,str):
                item = smiles2graph(item)

        if include_y:
            ys.append(item.y)
        item_new = Data()
        if isinstance(item,Data):
            for key in item.keys:
                item_new[key]=torch.tensor(item[key]) if (not isinstance(item[key],torch.Tensor))  else item[key]
        else:
            for key in item.keys():
                item_new[key]=torch.tensor(item[key]) if (not isinstance(item[key],torch.Tensor))  else item[key]

        if transform_in_collator:
            # try:
            item_new = graphormer_data_transform_tensor(item_new,rich_features)
            # except:
            #     print('graph data error')
            #     print(item)
            #     continue

        items_new.append(item_new)



    items=items_new
    items = [item for item in items if item is not None and item['x'].size(0) <= max_node]
    items = [
        (
            # item.idx,
            item.attn_bias,
            item.attn_edge_type,
            item.spatial_pos,
            item.in_degree,
            item.out_degree,
            item.x,
            item.edge_input[:, :, :multi_hop_max_dist, :],
            # item.y,
        )
        for item in items
    ]
    (
        # idxs,
        attn_biases,
        attn_edge_types,
        spatial_poses,
        in_degrees,
        out_degrees,
        xs,
        edge_inputs,
        # ys,
    ) = zip(*items)

    for idx, _ in enumerate(attn_biases):
        attn_biases[idx][1:, 1:][spatial_poses[idx] >= spatial_pos_max] = float("-inf")

    max_node_num = max(i.size(0) for i in xs)
    max_dist = max(i.size(-2) for i in edge_inputs)

    # y = torch.cat(ys)
    x = torch.cat([pad_2d_unsqueeze(i, max_node_num) for i in xs])

    # edge_input_padded=[pad_3d_unsqueeze(i, max_node_num, max_node_num, max_dist) for i in edge_inputs]
    # edge_input_ori = torch.cat(edge_input_padded
    # )
    edge_input = pad_3d_sequences(edge_inputs,max_node_num, max_node_num, max_dist)
    # assert (edge_input_ori-edge_input).abs().sum()==0

    attn_bias = torch.cat(
        [pad_attn_bias_unsqueeze(i, max_node_num + 1) for i in attn_biases]
    )

    attn_edge_type = torch.cat(
        [pad_edge_type_unsqueeze(i, max_node_num) for i in attn_edge_types]
    )

    spatial_pos = torch.cat(
        [pad_spatial_pos_unsqueeze(i, max_node_num) for i in spatial_poses]
    )

    in_degree = torch.cat([pad_1d_unsqueeze(i, max_node_num) for i in in_degrees])


    if include_y:
        y = torch.cat(ys)
        return Data(
            # idx=torch.LongTensor(idxs),
            attn_bias=attn_bias,
            attn_edge_type=attn_edge_type,
            spatial_pos=spatial_pos,
            in_degree=in_degree,
            out_degree=in_degree,  # for undirected graph
            x=x,
            edge_input=edge_input,
            y=y,
        )
    else:
        return Data(
            # idx=torch.LongTensor(idxs),
            attn_bias=attn_bias,
            attn_edge_type=attn_edge_type,
            spatial_pos=spatial_pos,
            in_degree=in_degree,
            out_degree=in_degree,  # for undirected graph
            x=x,
            edge_input=edge_input,
            # y=y,
        )



class CollatorForGraphormer(Collater):
    def __init__(self, **kwargs):
        self.transform_in_collator = kwargs.pop('transform_in_collator')
        self.include_y = kwargs.pop('include_y')
        self.rich_features = kwargs.pop('rich_features')
        try:
            super().__init__([],[])
        except:
            print('torch_geometric version is too low. Collater only accept one parameter for initialization')
            super().__init__([])

    def collate(self, batch):
        graph_batch = collator_graph_data(batch, transform_in_collator=self.transform_in_collator,
                                          include_y=self.include_y, rich_features=self.rich_features)
        return graph_batch


class CollaterForGraphormerMultiTask(CollatorForGraphormer):
    def collate(self, examples):
        graph_batch=[]
        labels_cla_batch=[]
        labels_reg_batch=[]
        for example_data in examples:
            graph_data=example_data['graph']
            graph_batch.append(graph_data)
            labels_cla_batch.append(example_data['label_cla'])
            labels_reg_batch.append(example_data['label_reg'])

        graph_batch = collator_graph_data(graph_batch, transform_in_collator=self.transform_in_collator,
                                          include_y=self.include_y, rich_features=self.rich_features)
        label_cla = torch.tensor(labels_cla_batch)
        label_reg = torch.tensor(labels_reg_batch)

        return {'graph':graph_batch,'label_cla':label_cla,'label_reg':label_reg}