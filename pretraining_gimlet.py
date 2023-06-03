import logging
import os
import sys
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional
import datasets
from datasets import load_dataset
import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoConfig,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    is_torch_tpu_available,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
from transformers import AutoTokenizer
from basic_pipeline import load_graph_args,eval_result
from model import GIMLETConfig,GinConfig,KVPLMConfig,MoMuConfig,MolT5Config,get_model,GalacticaConfig
from dataloaders import graph_text_tokenizer_dict,graph_text_collator_dict,graphormer_transform_for_dataset,WrapDataset
import torch.utils.data
import re
import numpy as np
from sklearn.metrics import (roc_auc_score,f1_score,confusion_matrix,r2_score)
from tqdm import tqdm
import faulthandler
import matplotlib.pyplot as plt

faulthandler.enable()

check_min_version("4.24.0.dev0")
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")
os.environ["WANDB_DISABLED"] = "true"

logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )

    transformer_backbone: str = field(default='')

    attention_fasion: str= field(default='sequential')

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated."
            )
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )
    line_by_line: bool = field(
        default=False,
        metadata={"help": "Whether distinct lines of text in the dataset are to be handled as distinct sequences."},
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    rich_features: Optional[bool] = field(default=False)
    transform_in_collator: Optional[bool] = field(default=False)
    wrap_dataset: Optional[bool] = field(default=False)

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                if extension not in ["csv", "json", "txt"]:
                    raise ValueError("`train_file` should be a csv, a json or a txt file.")
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                if extension not in ["csv", "json", "txt"]:
                    raise ValueError("`validation_file` should be a csv, a json or a txt file.")

# def eval_result(trainer,task_type='cla'):
#
#     loader = trainer.get_eval_dataloader()
#     model = trainer.model
#     device=model.device
#     tokenizer=trainer.tokenizer
#
#     str_y='Yes'
#     str_n='No'
#
#     id_y=tokenizer.convert_tokens_to_ids(tokenizer.tokenize(str_y))[0]
#     id_n=tokenizer.convert_tokens_to_ids(tokenizer.tokenize(str_n))[0]
#     # id_y=trainer.tokenizer(str_y)['input_ids'][0]
#     # id_n=trainer.tokenizer(str_n)['input_ids'][0]
#     id_invalid=-100
#
#
#     if task_type=='cla':
#         model.eval()
#         y_true, y_scores = [], []
#
#         for step, batch in enumerate(loader):
#             for key in batch.keys():
#                 batch[key] = batch[key].to(device)
#             with torch.no_grad():
#                 labels = batch["labels"]
#                 if hasattr(model, 'decoder'):
#                     del batch["labels"]
#                     batch["max_length"] = 3  # <PAD> CLASS <EOS>
#                     output = model.generate(
#                         **batch, output_scores=True, return_dict_in_generate=True
#                         # num_beams=beam_size,
#                         # no_repeat_ngram_size=no_repeat_ngram_size,
#                     )
#                     logits = output.scores[0].unsqueeze(1)  # logits of CLASS
#                 elif hasattr(model.base_model, 'decoder'): #galactica
#                     del batch["labels"]
#                     batch["max_new_tokens"] = 1  # <PAD> CLASS <EOS>
#                     # batch["max_length"] = 3
#                     output = model.generate(
#                         **batch, output_scores=True, return_dict_in_generate=True
#                         # num_beams=beam_size,
#                         # no_repeat_ngram_size=no_repeat_ngram_size,
#                     )
#                     # for i in range(output.sequences.shape[0]):
#                     #     print(tokenizer.decode(output.sequences[i]))
#                     logits = output.scores[0].unsqueeze(1)  # logits of CLASS
#                 elif hasattr(model,'graph_encoder'): #momu, contrastive model; the model code require label
#                     logits = model(**batch)['logits']
#                 else: #kvplm, self-masking model
#                     del batch["labels"]
#                     logits = model(**batch)['logits']
#             index = labels != id_invalid #mask both text not answer and invalid labels; shape: [batch,answer length]
#             if not isinstance(logits,dict): # for generative model
#                 # try:
#                 assert logits[index].ndim==2 # selected answer shape:[n_valid_sample,n_vocabulary]
#                 # except:
#                 #     print(batch['labels'])
#                 #     print(logits[index].shape)
#                 pred=(logits[index][:, id_y] - logits[index][:, id_n]).view([-1,1])
#                 true = labels[index].view(pred.shape)
#                 true[true == id_y] = 1
#                 true[true == id_n] = 0
#                 true[true == id_invalid] = -100
#             else: # for contrastive model, logits is dict
#                 pred = (logits['pos'].unsqueeze(1)[index] - logits['neg'].unsqueeze(1)[index]).view([-1, 1]) #shape of logits['pos] and logits['pos] are [batch]
#                 true = labels[index].view(pred.shape)
#                 assert torch.sum(true == id_invalid) == 0 # For contrastive model, invalid label is previously replaced by id_invalid(-100). Replace it here. Not necessary, because only valid label are selected
#
#             y_true.append(true)
#             y_scores.append(pred)
#
#         y_true = torch.cat(y_true, dim=0).cpu().numpy()
#         y_scores = torch.cat(y_scores, dim=0).cpu().numpy()
#         y_pred=(y_scores>0).astype(int)
#
#         roc_list = []
#         for i in range(y_true.shape[1]):
#             # AUC is only defined when there is at least one positive data.
#             if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
#                 is_valid = y_true[:, i]  >= 0
#                 roc_list.append(roc_auc_score(y_true[is_valid, i], y_scores[is_valid, i]))
#             else:
#                 print('{} is invalid'.format(i))
#
#         if len(roc_list) < y_true.shape[1]:
#             print(len(roc_list))
#             print('Some target is missing!')
#             print('Missing ratio: %f' % (1 - float(len(roc_list)) / y_true.shape[1]))
#
#         acc =  int((y_pred == y_true).sum()) / len(y_true)
#         auc = roc_list
#         f1 = f1_score(y_true,
#                       y_pred)
#         con_m = confusion_matrix(y_true,
#                       y_pred)
#         print('f1 ', f1)
#         print('confusion matrix ', con_m)
#         print('pred_rate ', y_pred.mean())
#         return {'acc': acc, 'auc': auc[0]}
#
#     else:
#         model.eval()
#         y_true, y_scores = [], []
#         for step, batch in enumerate(loader):
#             for key in batch.keys():
#                 batch[key] = batch[key].to(device)
#             with torch.no_grad():
#                 logits = model(**batch)['logits']
#             # index = batch['labels'] != id_invalid #mask both text not answer and invalid labels; shape: [batch,answer length]
#             if not isinstance(logits, dict):  # for generative model
#
#                 pred=[]
#                 for i in range(logits.shape[0]):
#                     ind_valid=batch['labels'][i, :]>0
#                     pred.append(tokenizer.decode(logits[i,ind_valid,:].argmax(1)))
#                 pred_number=[]
#                 for result in pred:
#                     number_list=re.findall(r"-?\d+\.?\d*e??\d*?",result)
#                     try:
#                         decoded_number=eval(number_list[0])
#                     except:
#                         decoded_number=float(np.nan)
#
#                     pred_number.append(decoded_number)
#                 true=[]
#                 for i in range(batch['labels'].shape[0]):
#                     true.append(tokenizer.decode(batch['labels'][i, batch['labels'][i, :]>0]))
#                 true_number=[]
#                 for result in true:
#                     number_list=re.findall(r"-?\d+\.?\d*e??\d*?",result)
#                     true_number.append(eval((number_list[0])) if len(number_list)>0 else float(np.nan))
#
#             else:  # for contrastive model, logits is dict
#                 raise ValueError("Not implemented for dict output!")
#
#             y_true+=true_number
#             y_scores+=pred_number
#
#         y_true = torch.tensor(y_true)
#         y_scores = torch.tensor(y_scores)
#
#         ind=(~y_scores.isnan())&(y_scores.abs()<reg_thre_by_task(args.dataset))
#         # ind = (~y_scores.isnan())
#         ratio=ind.float().mean()
#         y_true=y_true[ind]
#         y_scores=y_scores[ind]
#
#         mrs=(y_true-y_scores).std()
#         naive_msr=(y_true-y_true.mean()).std()
#
#         corrcoef=np.corrcoef(y_true,y_scores)[0,1]
#         r2=r2_score(y_true,y_scores)
#
#         plt.figure()
#         plt.scatter(y_true,y_scores)
#         global fig
#         plt.savefig('cache/{}fig{}.png'.format(args.dataset,fig))
#         fig+=1
#
#         print(naive_msr)
#
#         return {'ratio':float(ratio),'RMSE':float(mrs),'corrcoef':float(corrcoef),'R-Square':float(r2)}, 0, y_true, y_scores


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):

        model_args, data_args, training_args, extra_paras = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args,left = parser.parse_args_into_dataclasses(return_remaining_strings=True)

    model_args,graph_args=load_graph_args(model_args,left)

    send_example_telemetry("run_mlm", model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
            )
            raw_datasets["train"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
            )
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
            extension = data_args.train_file.split(".")[-1]
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
            extension = data_args.validation_file.split(".")[-1]
        if extension == "txt":
            extension = "text"
        raw_datasets = load_dataset(
            extension,
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
        # "do_lower_case": model_args.all_data_to_lower_case,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if model_args.model_name_or_path:
        if model_args.transformer_backbone !='':
            model=get_model(model_args, graph_args,tokenizer)

        else:
            config_kwargs = {
                "cache_dir": model_args.cache_dir,
                "revision": model_args.model_revision,
                "use_auth_token": True if model_args.use_auth_token else None,
            }
            if model_args.config_name:
                config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
            elif model_args.model_name_or_path:
                config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
            else:
                config = CONFIG_MAPPING[model_args.model_type]()
                logger.warning("You are instantiating a new config instance from scratch.")
                if model_args.config_overrides is not None:
                    logger.info(f"Overriding config: {model_args.config_overrides}")
                    config.update_from_string(model_args.config_overrides)
                    logger.info(f"New config: {config}")
            model = AutoModelForMaskedLM.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
            )
            model.resize_token_embeddings(len(tokenizer))

    else:
        config_kwargs = {
            "cache_dir": model_args.cache_dir,
            "revision": model_args.model_revision,
            "use_auth_token": True if model_args.use_auth_token else None,
        }
        if model_args.config_name:
            config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
        elif model_args.model_name_or_path:
            config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
        else:
            config = CONFIG_MAPPING[model_args.model_type]()
            logger.warning("You are instantiating a new config instance from scratch.")
            if model_args.config_overrides is not None:
                logger.info(f"Overriding config: {model_args.config_overrides}")
                config.update_from_string(model_args.config_overrides)
                logger.info(f"New config: {config}")
        logger.info("Training new model from scratch")
        model = AutoModelForMaskedLM.from_config(config)

    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
    else:
        column_names = raw_datasets["validation"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    if data_args.max_seq_length is None:
        max_seq_length = tokenizer.model_max_length
        if max_seq_length > 1024:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 1024 instead. You can change that default value by passing --max_seq_length xxx."
            )
            max_seq_length = 1024
    else:
        if data_args.max_seq_length > tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
                f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
            )
        max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    if data_args.line_by_line:
        # When using line_by_line, we just tokenize each nonempty line.
        padding = "max_length" if data_args.pad_to_max_length else False

        if model_args.transformer_backbone!='':
            tokenize_function=lambda x: graph_text_tokenizer_dict[model_args.transformer_backbone](examples=x, tokenizer=tokenizer, text_column_name=text_column_name, padding=padding, max_seq_length=max_seq_length, rich_features=data_args.rich_features, transform_in_collator=(data_args.transform_in_collator or data_args.wrap_dataset))
        else:
            def tokenize_function(examples):
                # Remove empty lines
                examples[text_column_name] = [
                    line for line in examples[text_column_name] if len(line) > 0 and not line.isspace()
                ]
                return tokenizer(
                    examples[text_column_name],
                    padding=padding,
                    truncation=True,
                    max_length=max_seq_length,
                    # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
                    # receives the `special_tokens_mask`.
                    return_special_tokens_mask=True,
                )
        # tokenize_function(raw_datasets['train'][0])
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(raw_datasets['train']), data_args.max_train_samples)
            raw_datasets['train'] = raw_datasets['train'].shuffle().select(range(max_train_samples))
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(raw_datasets['validation']), data_args.max_eval_samples)
            raw_datasets['validation'] = raw_datasets['validation'].shuffle().select(range(max_eval_samples))

        with training_args.main_process_first(desc="dataset map tokenization"):
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=False,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=[text_column_name],
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on dataset line_by_line",
            )

        if data_args.wrap_dataset:
            def transform_func(examples):
                return graphormer_transform_for_dataset(examples=examples,rich_features=data_args.rich_features)
            if training_args.do_train:
                tokenized_datasets["train"]=WrapDataset(data=tokenized_datasets["train"],transform=transform_func)
            if training_args.do_eval:
                tokenized_datasets["validation"] = WrapDataset(data=tokenized_datasets["validation"], transform=transform_func)

    else:
        def tokenize_function(examples):
            return tokenizer(examples[text_column_name], return_special_tokens_mask=True)

        with training_args.main_process_first(desc="dataset map tokenization"):
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on every text in dataset",
            )

        def group_texts(examples):
            # Concatenate all texts.
            concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])

            if total_length >= max_seq_length:
                total_length = (total_length // max_seq_length) * max_seq_length

            result = {
                k: [t[i : i + max_seq_length] for i in range(0, total_length, max_seq_length)]
                for k, t in concatenated_examples.items()
            }
            return result

        with training_args.main_process_first(desc="grouping texts together"):
            tokenized_datasets = tokenized_datasets.map(
                group_texts,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc=f"Grouping texts in chunks of {max_seq_length}",
            )

    if training_args.do_train:
        if "train" not in tokenized_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = tokenized_datasets["train"]

    if training_args.do_eval:
        if "validation" not in tokenized_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = tokenized_datasets["validation"]

        def preprocess_logits_for_metrics(logits, labels):
            if isinstance(logits, tuple):

                logits = logits[0]
            return logits.argmax(dim=-1)


    pad_to_multiple_of_8 = data_args.line_by_line and training_args.fp16 and not data_args.pad_to_max_length
    if model_args.transformer_backbone!='':
        data_collator = graph_text_collator_dict[model_args.transformer_backbone](
            tokenizer=tokenizer,
            mlm_probability=data_args.mlm_probability,
            pad_to_multiple_of=8 if pad_to_multiple_of_8 else None,
            transform_in_collator=data_args.transform_in_collator,
            rich_features=data_args.rich_features
        )
    else:
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm_probability=data_args.mlm_probability,
            pad_to_multiple_of=8 if pad_to_multiple_of_8 else None,
        )

    # Initialize our Trainer
    training_args.remove_unused_columns=False
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        # compute_metrics=compute_metrics if training_args.do_eval and not is_torch_tpu_available() else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
        if training_args.do_eval and not is_torch_tpu_available()
        else None
    )


    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        print('last_checkpoint: {}'.format(last_checkpoint))
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload
        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        str_y = 'Yes'
        str_n = 'No'
        label_dict={1:tokenizer.convert_tokens_to_ids(tokenizer.tokenize(str_y)),
                    0:tokenizer.convert_tokens_to_ids(tokenizer.tokenize(str_n)),
                    'invalid':[-100]}

        metrics = eval_result(trainer.model,trainer.get_eval_dataloader(),label_dict,trainer.tokenizer,'cla',model_args.transformer_backbone)
        metrics={'roc_auc': metrics[0]['score']}

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "fill-mask"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()