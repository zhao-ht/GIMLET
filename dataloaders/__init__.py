from .datasets_GPT import MoleculeDatasetGPT

from .molecule_datasets import (MoleculeDatasetRich,MoleculeDatasetSplitLabel)

from .graph_text_transform import gin_add_prompt_conditional_generation_transform_single, \
    \
    gimlet_add_prompt_conditional_generation_transform_single, \
    tokenize_function_gin_T5, tokenize_function_gimlet
from .graphormer_transform import graphormer_data_transform, graphormer_transform_for_dataset, \
    tokenize_function_graphormer_multitask

from .gin_text_collator import CollatorForGinTextLanguageModeling,CollatorForGNN
from .gimlet_collator import CollatorForGIMLETLanguageModeling
from .graphormer_collator import CollatorForGraphormer,CollaterForGraphormerMultiTask
from .galatica_smiles_collator import galactica_add_prompt_conditional_generation_transform_single,galactica_conditional_generation_tokenizer
from .gpt3_smiles_collator import gpt3_add_prompt_conditional_generation_transform_single, gpt3_conditional_generation_tokenizer
from .kvplm_smiles_collator import CollatorForKVPLM,kvplm_add_prompt_conditional_generation_transform_single,CollatorForSmilesTextLanguageModeling,kvplm_conditional_generation_tokenizer
from .momu_collator import CollatorForContrastiveGraphLanguageModeling, contrastive_add_prompt_conditional_generation_transform_single,contrastive_conditional_generation_tokenizer

from .wrap_dataset import WrapDataset


graph_text_tokenizer_dict={'gint5':tokenize_function_gin_T5, 'gimlet':tokenize_function_gimlet, 'kvplm':kvplm_conditional_generation_tokenizer, 'momu':contrastive_conditional_generation_tokenizer, 'galactica':galactica_conditional_generation_tokenizer, 'gpt3':gpt3_conditional_generation_tokenizer}

graph_text_collator_dict={'gint5':CollatorForGinTextLanguageModeling, 'gimlet':CollatorForGIMLETLanguageModeling, 'kvplm':CollatorForSmilesTextLanguageModeling, 'momu':CollatorForContrastiveGraphLanguageModeling, 'galactica':CollatorForSmilesTextLanguageModeling, 'gpt3':CollatorForSmilesTextLanguageModeling}

add_prompt_transform_dict={'gint5':gin_add_prompt_conditional_generation_transform_single,
                                                         'gimlet':gimlet_add_prompt_conditional_generation_transform_single,
                                                         'kvplm':kvplm_add_prompt_conditional_generation_transform_single,
                                                         'momu':contrastive_add_prompt_conditional_generation_transform_single,
                                                         'galactica':galactica_add_prompt_conditional_generation_transform_single,
                                                         'gpt3': gpt3_add_prompt_conditional_generation_transform_single}


GraphData_collator={'gnn':CollatorForGNN,'graphormer':CollatorForGraphormer,'kvplm':CollatorForKVPLM,'gimlet':CollatorForGraphormer}




