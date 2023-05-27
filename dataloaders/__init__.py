from .datasets_GPT import MoleculeDatasetGPT
from .molecule_3D_dataset import Molecule3DDataset
from .molecule_3D_masking_dataset import Molecule3DMaskingDataset
from .molecule_contextual_datasets import MoleculeContextualDataset
from .molecule_datasets import (MoleculeDatasetRich,MoleculeDatasetSplitLabel)
from .molecule_graphcl_dataset import MoleculeDataset_graphcl
from .molecule_graphcl_masking_dataset import MoleculeGraphCLMaskingDataset
# from .molecule_motif_datasets import RDKIT_PROPS, MoleculeMotifDataset
from .TextMoleculeDataset import TextMoleculeReplaceDataset


# graph_text_tokenizer includes data transforms for all the models, especially contains transforms with text tokenizer

from .graph_text_tokenizer import tokenize_function_gin_T5,tokenize_function_graphormer_T5,tokenize_function_graphormer_multitask,graphormer_transform_for_dataset

# data transforms are similiar to tokenizer functions, except they do not contain text tokenization process

from .graph_text_transform import graphormer_data_transform,gin_add_prompt_conditional_generation_transform_single,gin_add_prompt_conditional_generation_transform_sample,graphormer_add_prompt_conditional_generation_transform_single,graphormer_add_prompt_conditional_generation_transform_sample

# data collators may also includes data transforms when preprocessing data is not fessible (raise error)

from .gin_text_collator import CollatorForGinTextLanguageModeling,CollatorForGNN
from .graphormer_text_collator import CollatorForGraphormerTextLanguageModeling
from .graphormer_collator import CollatorForGraphormer,CollaterForGraphormerMultiTask
from .galatica_smiles_collator import galactica_add_prompt_conditional_generation_transform_single,galactica_conditional_generation_tokenizer
from .gpt3_smiles_collator import gpt3_add_prompt_conditional_generation_transform_single, gpt3_conditional_generation_tokenizer
from .kvplm_smiles_collator import CollatorForKVPLM,kvplm_add_prompt_conditional_generation_transform_sample,kvplm_add_prompt_conditional_generation_transform_single,CollatorForSmilesTextLanguageModeling,kvplm_conditional_generation_tokenizer
from .momu_collator import CollatorForContrastiveGraphLanguageModeling, contrastive_add_prompt_conditional_generation_transform_single,contrastive_add_prompt_conditional_generation_transform_sample,contrastive_conditional_generation_tokenizer
from torch_geometric.data.dataloader import Collater
from transformers import DataCollatorForLanguageModeling,DefaultDataCollator

from .wrap_dataset import WrapDataset


GraphTransformer_tokenizer_dict={'gin':{'t5':tokenize_function_gin_T5},'graphormer':{'t5':tokenize_function_graphormer_T5},'kvplm':{'kvplm':kvplm_conditional_generation_tokenizer},'momu':{'momu':contrastive_conditional_generation_tokenizer},'galactica':{'galactica':galactica_conditional_generation_tokenizer},'gpt3':{'gpt3':gpt3_conditional_generation_tokenizer}}

GraphTransformer_collator_dict={'gin':{'t5':CollatorForGinTextLanguageModeling},'graphormer':{'t5':CollatorForGraphormerTextLanguageModeling},'kvplm':{'kvplm':CollatorForSmilesTextLanguageModeling},'momu':{'momu':CollatorForContrastiveGraphLanguageModeling},'galactica':{'galactica':CollatorForSmilesTextLanguageModeling}, 'gpt3':{'gpt3':CollatorForSmilesTextLanguageModeling}}

add_prompt_conditional_generation_transform_single_dict={'gin':gin_add_prompt_conditional_generation_transform_single,
                                                         'graphormer':graphormer_add_prompt_conditional_generation_transform_single,
                                                         'kvplm':kvplm_add_prompt_conditional_generation_transform_single,
                                                         'momu':contrastive_add_prompt_conditional_generation_transform_single,
                                                         'galactica':galactica_add_prompt_conditional_generation_transform_single,
                                                         'gpt3': gpt3_add_prompt_conditional_generation_transform_single}

add_prompt_conditional_generation_transform_sample_dict={'gin':gin_add_prompt_conditional_generation_transform_sample,
                                                         'graphormer':graphormer_add_prompt_conditional_generation_transform_sample,
                                                         'kvplm':kvplm_add_prompt_conditional_generation_transform_sample,
                                                         'momu':contrastive_add_prompt_conditional_generation_transform_sample}

# GraphTransformer_collator_dict_add_prompt={'gin':{'t5':CollatorForGinTextLanguageModeling},'graphormer':{'t5':CollatorForGraphormerTextLanguageModeling},'kvplm':{'kvplm':CollatorForSmilesTextLanguageModeling},'momu':{'momu':CollatorForContrastiveGraphLanguageModeling}}

GraphData_collator={'gnn':CollatorForGNN,'graphormer':CollatorForGraphormer,'kvplm':CollatorForKVPLM,'grapht5':CollatorForGraphormer}




