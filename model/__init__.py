from .graphormer.models.graphormer import GraphormerModel
from .Arguments import GraphormerConfig,GinConfig,KVPLMConfig,MoMuConfig,MolT5Config, GalacticaConfig, GPT3Config
from .GraphT5TransformerForConditionalGeneration import GraphT5TransformerForConditionalGeneration
from .GraphT5TransformerForGraphOnly import GraphT5TransformerForGraphOnly
from .graphormer.models.graphormer import base_architecture,graphormer_base_architecture,graphormer_slim_architecture,graphormer_large_architecture
from .graphormer.models.graphormer_multitask import GraphormerModelMultiTask
from .KVPLM.kvplm_model import prepare_kvplm_model
from .MoMu.MoMu_Model import get_MoMu_model
from .llm import LLM
from transformers import OPTForCausalLM
import torch.optim as optim
from .molecule_gnn_model import GNN, GNN_graphpred
from transformers import (
    AutoTokenizer,
)

GraphTransformer_dict={'t5':GraphT5TransformerForConditionalGeneration,'kvplm':prepare_kvplm_model,'momu':get_MoMu_model, 'galactica':OPTForCausalLM}

Graphormer_version_dict={'graphormer':base_architecture,
                         'graphormer_base':graphormer_base_architecture,
                         'graphormer_slim':graphormer_slim_architecture,
                         'graphormer_large':graphormer_large_architecture}

from transformers import (
    AutoConfig,
)

import torch
import os
import openai



def get_model(args,graph_args,tokenizer):
    if not (args.graph_transformer_graph_backbone in ['kvplm','momu','galactica','gpt3']):
        config_kwargs = {
            "cache_dir": None,
            "revision": 'main',
            "use_auth_token":  None,
        }
        config = AutoConfig.from_pretrained(args.model_name_or_path, **config_kwargs)
        graph_args.graph_transformer_graph_backbone = args.graph_transformer_graph_backbone
        graph_args.attention_fasion = args.attention_fasion

        if graph_args.unimodel:
            ckpt=torch.load(os.path.join(args.model_name_or_path,'pytorch_model.bin'))
            if 'encoder.graph_encoder.encoder.graph_encoder.graph_node_feature.atom_encoder.weight' in ckpt:
                state_dict={}
                for key in ckpt.keys():
                    if not 'encoder.graph_encoder.encoder' in key:
                        state_dict[key]=ckpt[key]
                    elif 'graph_encoder.encoder.graph_encoder' in key:
                            key_new=key.replace('graph_encoder.encoder.graph_encoder','graph_encoder')
                            state_dict[key_new] = ckpt[key]
                    else:
                        pass
            else:
                state_dict=None
        else:
            state_dict=None

        model = GraphTransformer_dict[args.graph_transformer_text_backbone].from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            graph_args=graph_args,
            cache_dir=None,
            revision='main',
            use_auth_token=None,
            ignore_mismatched_sizes=True,
            state_dict=state_dict
        )
        model.resize_token_embeddings(len(tokenizer))
    elif args.graph_transformer_graph_backbone == 'kvplm':
        model = GraphTransformer_dict[args.graph_transformer_text_backbone](graph_args)
    elif args.graph_transformer_graph_backbone == 'momu':
        model = GraphTransformer_dict[args.graph_transformer_text_backbone](graph_args)
    elif args.graph_transformer_graph_backbone == 'galactica':
        model = GraphTransformer_dict[args.graph_transformer_text_backbone].from_pretrained(
            args.model_name_or_path,
        )
    elif args.graph_transformer_graph_backbone == 'gpt3':
        model = LLM(model=args.model_name_or_path)
    else:
        raise ValueError("not supported model type")
    return model


def get_graph_model(args,graph_args):
    if args.backbone == 'graphormer':
        graph_args.num_classes=args.num_tasks
        model = GraphormerModel.build_model(graph_args)
        # model_param_group = [{'params': model.molecule_model.parameters()},
        #                      {'params': model.graph_pred_linear.parameters(),
        #                       'lr': args.lr * args.lr_scale}]
        optimizer = optim.Adam(model.parameters(), lr=args.lr,
                               weight_decay=args.decay)
    elif args.backbone == 'grapht5':
        graph_args.num_classes=args.num_tasks

        config_kwargs = {
            "cache_dir": None,
            "revision": 'main',
            "use_auth_token":  None,
        }
        config = AutoConfig.from_pretrained(args.model_name_or_path, **config_kwargs)
        # graph_args.graph_transformer_graph_backbone = args.graph_transformer_graph_backbone
        # graph_args.attention_fasion = args.attention_fasion

        if graph_args.unimodel:
            ckpt=torch.load(os.path.join(args.model_name_or_path,'pytorch_model.bin'))
            if 'encoder.graph_encoder.encoder.graph_encoder.graph_node_feature.atom_encoder.weight' in ckpt:
                state_dict={}
                for key in ckpt.keys():
                    if not 'encoder.graph_encoder.encoder' in key:
                        state_dict[key]=ckpt[key]
                    elif 'graph_encoder.encoder.graph_encoder' in key:
                            key_new=key.replace('graph_encoder.encoder.graph_encoder','graph_encoder')
                            state_dict[key_new] = ckpt[key]
                    else:
                        pass
            else:
                state_dict=None
        else:
            state_dict=None

        model = GraphT5TransformerForGraphOnly.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            graph_args=graph_args,
            cache_dir=None,
            revision='main',
            use_auth_token=None,
            ignore_mismatched_sizes=True,
            state_dict=state_dict
        )

        tokenizer_kwargs = {
            "cache_dir": None,
            "use_fast": True,
            "revision": 'main',
            "use_auth_token": None,
        }
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, **tokenizer_kwargs)
        model.resize_token_embeddings(len(tokenizer))
        model_param_group = [{'params': model.encoder.parameters()},
                             {'params': model.decoder.parameters(),'lr':args.lr},
                             {'params':model.shared.parameters(),'lr':args.lr},
                             {'params':model.lm_head.parameters(),'lr':args.lr},
                             {'params': model.classifier.parameters(),
                              'lr': args.lr * args.lr_scale}]
        optimizer = optim.Adam(model_param_group, lr=args.lr,
                               weight_decay=args.decay)


    elif args.backbone == 'kvplm':
        graph_args.multi = args.num_tasks
        model = prepare_kvplm_model(graph_args)
        optimizer = optim.Adam(model.parameters(), lr=args.lr,
                               weight_decay=args.decay)
    else:
        assert args.backbone == 'gnn'
        molecule_model = GNN(**vars(graph_args))
        model = GNN_graphpred(args=graph_args, num_tasks=args.num_tasks,
                              molecule_model=molecule_model)
        model_param_group = [{'params': model.molecule_model.parameters()},
                             {'params': model.graph_pred_linear.parameters(),
                              'lr': args.lr * args.lr_scale}]
        optimizer = optim.Adam(model_param_group, lr=args.lr,
                               weight_decay=args.decay)
        if not (args.model_name_or_path == '' or args.model_name_or_path == 'None'):
            model.from_pretrained(args.model_name_or_path)
    return model,optimizer