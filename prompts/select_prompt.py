import pandas as pd
import os
import commentjson

def modify_name(name):
    name = name.replace('.ckpt', '.pt')
    name = name.replace('ckpts/', '')
    if name[-1] == '/':
        name = name[:-1]
    return name


def get_num_task(dataset):
    """ used in molecule_finetune.py """
    if dataset == 'tox21':
        return 12
    elif dataset in ['hiv', 'bace', 'bbbp', 'donor','esol','freesolv','lipo']:
        return 1
    elif dataset == 'pcba':
        return 128
    elif dataset == 'muv':
        return 17
    elif dataset == 'toxcast':
        return 617
    elif dataset == 'sider':
        return 27
    elif dataset == 'clintox':
        return 2
    elif dataset == 'cyp450':
        return 5
    raise ValueError(dataset + ': Invalid dataset name.')

augment_type_rename={'origin':'origin','detailed':'detail','rewrited':'rewrite','shortened':'shorten','expanded':'expand'}


#rename the augmented prompts

with open("../prompts_backup/downstream_task_prompt_multitask_new.json", 'r') as load_f:
    prompts_origin = commentjson.load(load_f)
with open("../prompts_backup/{}_downstream_task_prompt_multitask_new.json".format('augmented'),
          'r') as load_f:
    prompts_aug = commentjson.load(load_f)

if 'clintox' in prompts_origin:
    del prompts_origin['clintox']
if 'sider' in prompts_origin:
    del prompts_origin['sider']


prompts_rename_all={}

for task in prompts_origin.keys():
    prompts_rename_all[task]={}
    for split in prompts_origin[task].keys():
        if split not in prompts_rename_all[task]:
            prompts_rename_all[task][split]={}
        prompts_rename_all[task][split]['origin']=prompts_origin[task][split]
        if task in prompts_aug:
            for aug_type in prompts_aug[task][split].keys():
                prompts_rename_all[task][split][augment_type_rename[aug_type]]=prompts_aug[task][split][aug_type]
        else:
            print('{} not augmented'.format(task))

with open('prompt_downstream_task.json','w') as f:
    commentjson.dump(prompts_origin,f,indent=2)
with open('augmented_prompt_downstream_task.json','w') as f:
    commentjson.dump(prompts_rename_all,f,indent=2)



with open("../prompts_backup/ablated_downstream_task_prompt_multitask_new.json", 'r') as load_f:
    prompts_ablated = commentjson.load(load_f)
if 'clintox' in prompts_ablated:
    del prompts_ablated['clintox']
if 'sider' in prompts_ablated:
    del prompts_ablated['sider']
with open('ablated_prompt_downstream_task.json','w') as f:
    commentjson.dump(prompts_ablated,f,indent=2)


with open("../prompts_backup/downstream_task_prompt_multitask_new.json", 'r') as load_f:
    prompts = commentjson.load(load_f)

prompts_selected={}

for dataset in prompts:

    file_name = os.path.join('..','cache', 'result_max_prompt_table.csv')
    prompts_pd = pd.read_csv(file_name, index_col='unique_task_id')
    rename_keys = {}
    for name in prompts_pd.columns:
        rename_keys[name] = modify_name(name)
    prompts_pd = prompts_pd.rename(columns=rename_keys)
    prompt_dataset = {}
    model_name = modify_name('ckpts/test-mlmv-split_0_merge_split4_1_1x4_negative05_1_025x4.csv_maskt2g_sentence_t5_true_1/checkpoint-120000')
    for ind in range(get_num_task(dataset)):
        if dataset + '@' + str(ind) in prompts_pd.index.values:
            res = prompts_pd.loc[dataset + '@' + str(ind), model_name]
            if pd.isna(res):
                continue
            prompt_dataset[str(ind)] = [res]
        else:
            if str(ind) in prompts[dataset]:
                prompt_dataset[str(ind)] = [prompts[dataset][str(ind)][0]]
    prompts_selected[dataset]=prompt_dataset


if 'clintox' in prompts_selected:
    del prompts_selected['clintox']
if 'sider' in prompts_selected:
    del prompts_selected['sider']




file_name = os.path.join('..','cache', 'result_max_prompt_table.csv')
prompts_ref = pd.read_csv(file_name, index_col='unique_task_id')
rename_keys = {}
for name in prompts_ref.columns:
    name_new = modify_name(name)
    rename_keys[name] = name_new
prompts_ref = prompts_ref.rename(columns=rename_keys)

# splited_name = args.model_name_or_path.split('/')
# model_name=splited_name[-1] if len(splited_name[-1])>0 else splited_name[-2]

model_name = modify_name('ckpts/test-mlmv-split_0_merge_split4_1_1x4_negative05_1_025x4.csv_maskt2g_sentence_t5_true_1/checkpoint-120000')
prompts_ref = prompts_ref[model_name]



with open("../prompts_backup/downstream_task_prompt_multitask_new.json", 'r') as load_f:
    prompts_origin = commentjson.load(load_f)
with open("../prompts_backup/{}_downstream_task_prompt_multitask_new.json".format('augmented'),
          'r') as load_f:
    prompts_aug = commentjson.load(load_f)

prompt_aug_ref_selected = {}
for augment_type in ['rewrited', 'expanded', 'detailed', 'shortened']:
    for task_id, prompt in prompts_ref.iteritems():
        if not pd.isna(prompt):
            dataset, ind = task_id.split('@')
            if dataset in prompts_aug:
                id = prompts_origin[dataset][ind].index(prompt)
                prompt_new = prompts_aug[dataset][ind][augment_type][id]
                if dataset not in prompt_aug_ref_selected:
                    prompt_aug_ref_selected[dataset]={}
                    prompt_aug_ref_selected[dataset][ind]={augment_type_rename[augment_type]:[prompt_new]}
                elif ind not in prompt_aug_ref_selected[dataset]:
                    prompt_aug_ref_selected[dataset][ind] = {augment_type_rename[augment_type]:[prompt_new]}
                else:
                    assert augment_type_rename[augment_type] not in prompt_aug_ref_selected[dataset][ind]
                    prompt_aug_ref_selected[dataset][ind][augment_type_rename[augment_type]]=[prompt_new]
            else:
                print('{} not in prompts_aug'.format(task_id))

if 'clintox' in prompt_aug_ref_selected:
    del prompt_aug_ref_selected['clintox']
if 'sider' in prompt_aug_ref_selected:
    del prompt_aug_ref_selected['sider']



for dataset in prompts_selected.keys():
    for ind in prompts_selected[dataset].keys():
        if dataset in prompt_aug_ref_selected:
            if ind in prompt_aug_ref_selected[dataset]:
                prompt_aug_ref_selected[dataset][ind]['origin']=prompts_selected[dataset][ind]
            else:
                prompt_aug_ref_selected[dataset][ind]={}
                prompt_aug_ref_selected[dataset][ind]['origin'] = prompts_selected[dataset][ind]
        else:
            prompt_aug_ref_selected[dataset]={}
            prompt_aug_ref_selected[dataset][ind]={}
            prompt_aug_ref_selected[dataset][ind]['origin']=prompts_selected[dataset][ind]


with open('selected_prompt_downstream_task.json.json','w') as f:
    commentjson.dump(prompts_selected,f,indent=2)

with open('selected_augmented_prompt_downstream_task.json','w') as f:
    commentjson.dump(prompt_aug_ref_selected,f,indent=2)









all_prompt_pretrain={}

with open("../prompts_backup/prompt_T0.json", 'r') as load_f:
    all_prompt_pretrain['chembl_property'] = commentjson.load(load_f)

with open("../prompts_backup/prompt_assay.json", 'r') as load_f:
    all_prompt_pretrain['chembl'] = commentjson.load(load_f)

all_augmented_prompt_pretrain={}

with open("../prompts_backup/augmented_prompt_T0.json", 'r') as load_f:
    all_augmented_prompt_pretrain['chembl_property'] = commentjson.load(load_f)

with open("../prompts_backup/augmented_prompt_assay.json", 'r') as load_f:
    all_augmented_prompt_pretrain['chembl'] = commentjson.load(load_f)


all_augmented_prompt_pretrain_rename={}

for dataset in all_augmented_prompt_pretrain.keys():
    for ind in all_augmented_prompt_pretrain[dataset].keys():
        for augment_type in all_augmented_prompt_pretrain[dataset][ind].keys():
            if dataset not in all_augmented_prompt_pretrain_rename:
                all_augmented_prompt_pretrain_rename[dataset]={}
            if ind not in all_augmented_prompt_pretrain_rename[dataset]:
                all_augmented_prompt_pretrain_rename[dataset][ind]={}
            if augment_type_rename[augment_type] not in all_augmented_prompt_pretrain_rename[dataset][ind]:
                all_augmented_prompt_pretrain_rename[dataset][ind][augment_type_rename[augment_type]]=all_augmented_prompt_pretrain[dataset][ind][augment_type]



for dataset in all_prompt_pretrain.keys():
    for ind in all_prompt_pretrain[dataset].keys():
        all_augmented_prompt_pretrain_rename[dataset][ind]['origin']=all_prompt_pretrain[dataset][ind]

with open('prompt_pretrain.json','w') as f:
    commentjson.dump(all_prompt_pretrain,f,indent=2)

with open('augmented_prompt_pretrain.json','w') as f:
    commentjson.dump(all_augmented_prompt_pretrain_rename,f,indent=2)
