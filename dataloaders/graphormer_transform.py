from ogb.utils import smiles2graph
import torch
import numpy as np
# from . import algos

import scipy
import setuptools
import pyximport

pyximport.install(setup_args={"include_dirs": np.get_include()})
from . import algos
import numbers
# @torch.jit.script
def convert_to_single_emb(x, offset: int = 512):
    feature_num = x.shape[1] if len(x.shape) > 1 else 1
    feature_offset = (1 + np.arange(0, feature_num * offset, offset)).astype("long")
    x = x + feature_offset
    return x

#graphormer_data_transform is for numpy data, and graphormer_data_transform_tensor is for tensor data

def graphormer_data_transform(item,rich_features=False):
    #input item is standard graph data get by smiles2graph
    if 'node_feat' in item and 'edge_feat' in item:
        edge_attr, edge_index, x = item['edge_feat'], item['edge_index'], item['node_feat']
    elif hasattr(item, 'x') and hasattr(item, 'edge_attr'):
        edge_attr, edge_index, x = item.edge_attr, item.edge_index, item.x
    else:
        raise ValueError('item does not have expected keys or properties')

    if not rich_features:
        x = x[:,0:2]
        edge_attr = edge_attr[:,0:2]

    N = x.shape[0]
    x = convert_to_single_emb(x)

    # node adj matrix [N, N] bool
    adj = np.zeros([N, N]).astype(bool)
    adj[edge_index[0, :], edge_index[1, :]] = True

    # edge feature here
    if len(edge_attr.shape) == 1:
        edge_attr = edge_attr[:, None]
    attn_edge_type = np.zeros([N, N, edge_attr.shape[-1]]).astype('long')
    attn_edge_type[edge_index[0, :], edge_index[1, :]] = (
        convert_to_single_emb(edge_attr) + 1
    )

    shortest_path_result, path = algos.floyd_warshall(adj)
    # shortest_path_result2, path2 = scipy.sparse.csgraph.floyd_warshall(adj,return_predecessors=True)
    # shortest_path_result3, path3 = algos2.floyd_warshall(adj)
    max_dist = np.amax(shortest_path_result)
    edge_input = algos.gen_edge_input(max_dist, path, attn_edge_type)
    # edge_input2 = algos2.gen_edge_input(max_dist, path, attn_edge_type)
    spatial_pos = shortest_path_result.astype('long')
    attn_bias = np.zeros([N + 1, N + 1]).astype('float')  # with graph token

    # combine

    #Data type will also be transformed into dict type in cache
    # item_new=Data()
    item_new = {}
    item_new['x'] = x
    item_new['attn_bias'] = attn_bias
    item_new['attn_edge_type'] = attn_edge_type
    item_new['spatial_pos'] = spatial_pos
    item_new['in_degree'] = adj.astype('long').sum(1)
    item_new['out_degree'] = item_new['in_degree']  # for undirected graph
    item_new['edge_input'] = edge_input.astype('long')

    return item_new


def graphormer_data_transform_tensor(item,rich_features=False):
    if 'node_feat' in item and 'edge_feat' in item:
        edge_attr, edge_index, x = item['edge_feat'].long(), item['edge_index'].long(), item['node_feat'].long()
    elif hasattr(item, 'x') and hasattr(item, 'edge_attr'):
        edge_attr, edge_index, x = item.edge_attr.long(), item.edge_index.long(), item.x.long()
    else:
        raise ValueError('item does not have expected keys or properties')

    if not rich_features:
        x = x[:,0:2]
        edge_attr = edge_attr[:,0:2]


    N = x.size(0)
    x = convert_to_single_emb(x)

    # node adj matrix [N, N] bool
    adj = torch.zeros([N, N], dtype=torch.bool)
    adj[edge_index[0, :], edge_index[1, :]] = True

    # edge feature here
    if len(edge_attr.size()) == 1:
        edge_attr = edge_attr[:, None]
    attn_edge_type = torch.zeros([N, N, edge_attr.size(-1)], dtype=torch.long)
    attn_edge_type[edge_index[0, :], edge_index[1, :]] = (
        convert_to_single_emb(edge_attr) + 1
    )

    shortest_path_result, path = algos.floyd_warshall(adj.numpy())
    max_dist = np.amax(shortest_path_result)
    edge_input = algos.gen_edge_input(max_dist, path, attn_edge_type.numpy())
    spatial_pos = torch.from_numpy((shortest_path_result)).long()
    attn_bias = torch.zeros([N + 1, N + 1], dtype=torch.float)  # with graph token

    # combine
    item.x = x
    item.attn_bias = attn_bias
    item.attn_edge_type = attn_edge_type
    item.spatial_pos = spatial_pos
    item.in_degree = adj.long().sum(dim=1).view(-1)
    item.out_degree = item.in_degree  # for undirected graph
    item.edge_input = torch.from_numpy(edge_input).long()

    return item


def graphormer_transform_for_dataset(examples,rich_features):
    graph_data = smiles2graph(examples['graph'])
    graph_data = graphormer_data_transform(graph_data, rich_features)
    examples['graph']=graph_data
    return examples


def tokenize_function_graphormer_multitask(examples,rich_features,transform_in_collator):

    graph_data = smiles2graph(examples['graph'])

    # from pretrain_datasets.ChEMBL_STRING.mole_key import key_int, key_float

    label_cla=[]
    label_reg=[]
    for key in examples.keys():
        if len(key)>=4 and key[0:4]=='cla_':
            label_cla.append(examples[key])
        elif len(key)>=4 and key[0:4]=='reg_':
            label_reg.append(examples[key])

    if not transform_in_collator:
        graph_data = graphormer_data_transform(graph_data,rich_features)

    return {'graph': graph_data,
            'label_cla': label_cla,
            'label_reg': label_reg,}