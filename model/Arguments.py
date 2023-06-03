from dataclasses import dataclass, field
from typing import Optional

@dataclass
class GraphormerConfig:
    # dataset_name: str = field(
    #     default="pcqm4m",
    #     metadata={"help": "name of the dataset"},
    # )
    # num_classes
    num_classes: int = field(
        default=-1,
        metadata={"help": "number of classes or regression targets"},
    )

    max_nodes: int = field(
        default=128,
        metadata={"help": "max nodes per graph"},
    )

    # dataset_source: str = field(
    #     default="pyg",
    #     metadata={"help": "source of graph dataset, can be: pyg, dgl, ogb, smiles"},
    # )

    num_atoms: int = field(
        default=512 * 9,
        metadata={"help": "number of atom types in the graph"},
    )

    num_edges: int = field(
        default=512 * 3,
        metadata={"help": "number of edge types in the graph"},
    )

    num_in_degree: int = field(
        default=512,
        metadata={"help": "number of in degree types in the graph"},
    )

    num_out_degree: int = field(
        default=512,
        metadata={"help": "number of out degree types in the graph"},
    )

    num_spatial: int = field(
        default=512,
        metadata={"help": "number of spatial types in the graph"},
    )

    num_edge_dis: int = field(
        default=128,
        metadata={"help": "number of edge dis types in the graph"},
    )

    multi_hop_max_dist: int = field(
        default=5,
        metadata={"help": "max distance of multi-hop edges"},
    )

    spatial_pos_max: int = field(
        default=1024,
        metadata={"help": "max distance of multi-hop edges"},
    )

    edge_type: str = field(
        default="multi_hop",
        metadata={"help": "edge type in the graph"},
    )

    # seed:  int = field(default=0)

    # pretrained_model_name: str = field(
    #     default="none",
    #     metadata={"help": "name of used pretrained model"},
    # )

    not_load_pretrained_model_output_layer: bool = field(
        default=False,
        metadata={"help": "whether to load the output layer of pretrained model"},
    )

    arch: str = field(default='graphormer_base')

    graphonly_problem_type: str = field(default='')

    graphonly_readout: str= field(default='mean')

    restore_file_graphormer: str= field(default=None)

    unimodel: bool = field(default=False)

    maskt2g: bool = field(default=False)

    loss_reduction_method: str = field(default='token')


    # train_epoch_shuffle: bool = field(
    #     default=False,
    #     metadata={"help": "whether to shuffle the dataset at each epoch"},
    # )

    # user_data_dir: str = field(
    #     default="",
    #     metadata={"help": "path to the module of user-defined dataset"},
    # )

@dataclass
class GIMLETConfig:
    # dataset_name: str = field(
    #     default="pcqm4m",
    #     metadata={"help": "name of the dataset"},
    # )
    # num_classes
    num_classes: int = field(
        default=-1,
        metadata={"help": "number of classes or regression targets"},
    )

    max_nodes: int = field(
        default=128,
        metadata={"help": "max nodes per graph"},
    )

    # dataset_source: str = field(
    #     default="pyg",
    #     metadata={"help": "source of graph dataset, can be: pyg, dgl, ogb, smiles"},
    # )

    num_atoms: int = field(
        default=512 * 9,
        metadata={"help": "number of atom types in the graph"},
    )

    num_edges: int = field(
        default=512 * 3,
        metadata={"help": "number of edge types in the graph"},
    )

    num_in_degree: int = field(
        default=512,
        metadata={"help": "number of in degree types in the graph"},
    )

    num_out_degree: int = field(
        default=512,
        metadata={"help": "number of out degree types in the graph"},
    )

    num_spatial: int = field(
        default=512,
        metadata={"help": "number of spatial types in the graph"},
    )

    num_edge_dis: int = field(
        default=128,
        metadata={"help": "number of edge dis types in the graph"},
    )

    multi_hop_max_dist: int = field(
        default=5,
        metadata={"help": "max distance of multi-hop edges"},
    )

    spatial_pos_max: int = field(
        default=1024,
        metadata={"help": "max distance of multi-hop edges"},
    )

    edge_type: str = field(
        default="multi_hop",
        metadata={"help": "edge type in the graph"},
    )

    # seed:  int = field(default=0)

    # pretrained_model_name: str = field(
    #     default="none",
    #     metadata={"help": "name of used pretrained model"},
    # )
    dropout: float = field(default=0.0)

    encoder_embed_dim: int = field(
        default=0,
    )

    encoder_attention_heads: int = field(
        default=0,
    )

    encoder_layers: int = field(
        default=0,
    )

    encoder_normalize_before: bool = field(
        default=False,
    )
    apply_graphormer_init: bool = field(
        default=True,
    )

    graphonly_problem_type: str = field(default='')

    graphonly_readout: str= field(default='mean')

    maskt2g: bool = field(default=True)

    loss_reduction_method: str = field(default='token')


    # train_epoch_shuffle: bool = field(
    #     default=False,
    #     metadata={"help": "whether to shuffle the dataset at each epoch"},
    # )

    # user_data_dir: str = field(
    #     default="",
    #     metadata={"help": "path to the module of user-defined dataset"},
    # )



@dataclass
class GinConfig:
    # dataset_name: str = field(
    #     default="pcqm4m",
    #     metadata={"help": "name of the dataset"},
    # )
    # num_classes
    num_layer:int = field(default=5)
    emb_dim: int  = field(default=300)
    JK : str  = field(default= 'last')
    drop_ratio : float  = field(default= 0.5)
    gnn_type : str  = field(default= 'gin')

    loss_reduction_method: str = field(default='token')
    graph_pooling : str = field(default='mean')

    # train_epoch_shuffle: bool = field(
    #     default=False,
    #     metadata={"help": "whether to shuffle the dataset at each epoch"},
    # )

    # user_data_dir: str = field(
    #     default="",
    #     metadata={"help": "path to the module of user-defined dataset"},
    # )



@dataclass
class KVPLMConfig:
#     parser.add_argument("--config_file", default='/mnt/data/zhaohaiteng/KV-PLM/bert_base_config.json', type=str, )
#     parser.add_argument("--num_labels", default=2, type=int, )
#     parser.add_argument("--init_checkpoint", default=None, type=str, )
#     parser.add_argument("--task", default='tox21', type=str, )
#     parser.add_argument("--multi", default=1, type=int, )
#     parser.add_argument("--tok", default=0, type=int, )
#     parser.add_argument("--rx", default=0, type=int, )
#     parser.add_argument("--sm_pth", default='MoleculeNet/sm_', type=str, )
#     parser.add_argument("--resume", default=-1, type=int, )
#     parser.add_argument("--weight_decay", default=0, type=float, )
#     parser.add_argument("--lr", default=5e-6, type=float, )
#     parser.add_argument("--warmup", default=0.2, type=float, )
#     parser.add_argument("--total_steps", default=1200, type=int, )
#     parser.add_argument("--pth_data", default='MoleculeNet/sm_', type=str, )
#     parser.add_argument("--pth_lab", default='MoleculeNet/lab_', type=str, )
#     parser.add_argument("--pth_text", default='MoleculeNet/text_', type=str, )
#     parser.add_argument("--batch_size", default=64, type=int, )
#     parser.add_argument("--epoch", default=20, type=int, )
#     parser.add_argument("--seed", default=0, type=int, )
#     parser.add_argument("--output", default='/mnt/data/zhaohaiteng/KV-PLM/finetune_save/ckpt_test1', type=str, )

    config_file: str = field(default='ckpts/bert_base_config.json')
    num_labels: int = field(default=2)
    init_checkpoint: str = field(default=None)
    # multi: int = field(default=1)
    rx: int = field(default=0)
    # kvplm_language_model: bool = field(default=False)

    # task: str = field()


@dataclass
class MoMuConfig:
    # parser.add_argument("--device", default="0", type=str, )
    # parser.add_argument("--init_checkpoint", default="all_checkpoints/MoMu-S.ckpt", type=str, )
    # parser.add_argument("--output", default='finetune_save/sent_MoMu-S_73.pt', type=str, )
    # parser.add_argument("--data_type", default=0, type=int)  # 0-para, 1-sent
    # parser.add_argument("--if_test", default=1, type=int)
    # parser.add_argument("--if_zeroshot", default=1, type=int)
    # parser.add_argument("--pth_train", default='data/kv_data/train', type=str, )
    # parser.add_argument("--pth_dev", default='data/kv_data/dev', type=str, )
    # parser.add_argument("--pth_test", default='data/phy_data', type=str, )
    # parser.add_argument("--weight_decay", default=0, type=float, )
    # parser.add_argument("--lr", default=5e-5, type=float, )  # 4
    # parser.add_argument("--warmup", default=0.2, type=float, )
    # parser.add_argument("--total_steps", default=5000, type=int, )  # 3000
    # parser.add_argument("--batch_size", default=64, type=int, )
    # parser.add_argument("--epoch", default=30, type=int, )
    # parser.add_argument("--seed", default=73, type=int, )  # 73 99 108
    # parser.add_argument("--graph_aug", default='noaug', type=str, )
    # parser.add_argument("--text_max_len", default=128, type=int, )
    # parser.add_argument("--margin", default=0.2, type=int, )

    init_checkpoint: str =field(default=None)
    graph_aug: bool = field(default=True)


@dataclass
class MolT5Config:
    init_checkpoint: str = field(default='laituan245/molt5-base') 
    
    
@dataclass
class GalacticaConfig:
    init_checkpoint: str = field(default='facebook/galactica-125m')
     

@dataclass
class GPT3Config:
    init_checkpoint: str = field(default='text-davinci-003')
    max_tokens: int = field(default= 100)
    temperature: float = field(default=0.0)
     
      