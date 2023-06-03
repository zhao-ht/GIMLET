from .graphormer.models.graphormer import GraphormerModel
from .Arguments import GIMLETConfig,GraphormerConfig,GinConfig,KVPLMConfig,MoMuConfig,MolT5Config, GalacticaConfig, GPT3Config
from model.GIMLET.GIMLETTransformerForConditionalGeneration import GraphT5TransformerForConditionalGeneration
from .graphormer.models.graphormer import base_architecture,graphormer_base_architecture,graphormer_slim_architecture,graphormer_large_architecture
from .graphormer.models.graphormer_multitask import GraphormerModelMultiTask
from .KVPLM.kvplm_model import prepare_kvplm_model
from .MoMu.MoMu_Model import get_MoMu_model
from .llm import LLM
from transformers import OPTForCausalLM
import torch.optim as optim
from model.GIMLET.gnn_model import GNN, GNN_graphpred

GraphTransformer_dict={'gimlet':GraphT5TransformerForConditionalGeneration,'gint5':GraphT5TransformerForConditionalGeneration,'kvplm':prepare_kvplm_model,'momu':get_MoMu_model, 'galactica':OPTForCausalLM}



from transformers import (
    AutoConfig,
)


def get_model(args,graph_args,tokenizer):
    if not (args.transformer_backbone in ['kvplm','momu','galactica','gpt3']):
        config_kwargs = {
            "cache_dir": None,
            "revision": 'main',
            "use_auth_token":  None,
        }
        config = AutoConfig.from_pretrained(args.tokenizer_name, **config_kwargs)
        config.vocab_size=len(tokenizer)
        graph_args.transformer_backbone = args.transformer_backbone

        model = GraphTransformer_dict[args.transformer_backbone].from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            graph_args=graph_args,
            cache_dir=None,
            revision='main',
            use_auth_token=None,
            ignore_mismatched_sizes=True,
        )
        model.resize_token_embeddings(len(tokenizer))
    elif args.transformer_backbone == 'kvplm':
        model = GraphTransformer_dict[args.transformer_backbone](graph_args)
    elif args.transformer_backbone == 'momu':
        model = GraphTransformer_dict[args.transformer_backbone](graph_args)
    elif args.transformer_backbone == 'galactica':
        model = GraphTransformer_dict[args.transformer_backbone].from_pretrained(
            args.model_name_or_path,
        )
    elif args.transformer_backbone == 'gpt3':
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