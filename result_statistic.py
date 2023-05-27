import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly
import os
import argparse
import commentjson
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--result_file_list",type=str,nargs='+')
parser.add_argument("--output_result",action="store_true")
parser.add_argument("--output_prefix",type=str,default='')
parser.add_argument('--plot_figure',action='store_true')
parser.add_argument('--few_shot',action='store_true')
parser.add_argument('--augment_type',type=str,default='',help='do not set value; auto detected in code')
parser.add_argument('--augment',action='store_true')
parser.add_argument('--prompt_augmentation_file_prefix',default='augmented')
parser.add_argument('--refer_best_result',type=str,default='',help='when multi prompt is tested, this arg specify which prompt to choose')
args = parser.parse_args()

if args.output_prefix!='':
    args.output_prefix='_'+args.output_prefix

def task_type(dataset):
    if dataset in ['esol','freesolv','lipo']:
        return 'reg'
    else:
        return 'cla'


def model_name_replace(name):
    return name.replace('.ckpt','.pt')


def modify_name(name):
    name = name.replace('.ckpt', '.pt')
    name = name.replace('ckpts/', '')
    if name[-1] == '/':
        name = name[:-1]
    return name

for result_file in tqdm(args.result_file_list):

    try:
        result_origin = pd.read_csv(os.path.join('cache', result_file), header=None)
    except:
        result_origin=pd.read_csv(os.path.join('cache',result_file),header=None, encoding = 'gb18030')

    if 'few_shot' in result_file:
        args.few_shot=True
        args.augment_type=eval(result_file.split('_')[3])
        args.output_prefix='_fewshot'

    result_origin.columns =['None','dataset','split', 'model_name_or_path','epoch','lr','runseed','best_val_idx','train_roc','val_roc','test_roc','prompt'
                                               ] if args.few_shot else ['None','dataset', 'split','model_name_or_path', 'train_roc', 'val_roc', 'test_roc', 'prompt']

    result_origin['model_name_or_path'] = result_origin['model_name_or_path'].apply(model_name_replace)

    if args.augment:
        args.augment_type = result_file.split('_')[2]
        assert args.augment_type in ['rewrited', 'expanded', 'detailed', 'shortened','name']
    if args.refer_best_result!='':
        file_name=os.path.join('cache','result_max_prompt_table.csv')
        prompts_ref = pd.read_csv(file_name,index_col='unique_task_id')
        rename_keys={}
        for name in prompts_ref.columns:
            name_new=modify_name(name)
            rename_keys[name]=name_new
        prompts_ref=prompts_ref.rename(columns=rename_keys)

        # splited_name = args.model_name_or_path.split('/')
        # model_name=splited_name[-1] if len(splited_name[-1])>0 else splited_name[-2]

        model_name=modify_name(args.refer_best_result)
        prompts_ref=prompts_ref[model_name]

        with open("prompts_backup/downstream_task_prompt_multitask_new.json", 'r') as load_f:
            prompts_origin = commentjson.load(load_f)
        with open("prompts/{}_downstream_task_prompt_multitask_new.json".format(args.prompt_augmentation_file_prefix), 'r') as load_f:
            prompts_aug = commentjson.load(load_f)
        prompt_aug_ref=[]
        for task_id,prompt in prompts_ref.iteritems():
            if not pd.isna(prompt):
                dataset,ind=task_id.split('@')
                if dataset in prompts_aug:
                    id=prompts_origin[dataset][ind].index(prompt)
                    prompt_new=prompts_aug[dataset][ind][args.augment_type][id]
                    prompt_aug_ref.append([task_id,prompt_new])
                else:
                    print('{} not in prompts_aug'.format(task_id))
        prompt_aug_ref=pd.DataFrame(prompt_aug_ref,columns=['unique_task_id','prompt'])
        prompt_aug_ref=prompt_aug_ref.set_index('unique_task_id')

        result_origin=pd.merge(result_origin, prompt_aug_ref, on=['prompt'])

    if args.few_shot:
        args.task_type='few_shot'
    elif args.augment:
        args.task_type='augment'
    else:
        args.task_type='normal'



    models_key=set(result_origin['model_name_or_path'])
    result_per_task=[]

    models_key_names={}
    for model in models_key:
        if 'test-mlmv' in model:
            name_grapht0=model
            models_key_names[name_grapht0]='GraphT0'
        elif 'KV' in model:
            name_kvplm=model
            models_key_names[name_kvplm] = 'KVPLM'
        elif 'scibert' in model:
            name_momu=model
            models_key_names[name_momu] = 'MoMu'

    for model in models_key:
        result=result_origin[result_origin['model_name_or_path']==model]
        datasets_key=set(result['dataset'])
        for dataset in datasets_key:
            result_dataset=result[result['dataset']==dataset]
            split_keys=set(result_dataset['split'])
            for split in split_keys:
                result_split=result_dataset[result_dataset['split']==split]
                prompt_keys=set(result_split['prompt'])
                # assert len(prompt_keys)==len(result_split) #make sure no prompt is test twice
                mean_task=result_split[['train_roc','val_roc','test_roc']].mean()
                max_task = result_split[['train_roc', 'val_roc', 'test_roc']].max() if task_type(dataset)=='cla' else result_split[['train_roc', 'val_roc', 'test_roc']].min()

                try:
                    max_prompt=result_split.loc[result_split['test_roc'].idxmax()]['prompt'] if task_type(dataset)=='cla' else result_split.loc[result_split['test_roc'].idxmin()]['prompt']
                    max_abs_prompt=result_split.loc[np.abs(result_split['test_roc']-0.5).idxmax()]['prompt'] if task_type(dataset)=='cla' else ''
                except:
                    pass

                result_per_task.append([model,args.task_type,args.augment_type, dataset, split,str(dataset)+'@'+str(split), *mean_task,*max_task,max_prompt,max_abs_prompt])

    result_per_task=pd.DataFrame(result_per_task,columns=['model','task_type','augment_type', 'dataset', 'split','unique_task_id','train_mean','val_mean','test_mean','train_max','val_max','test_max','max_prompt','max_abs_prompt'])
    models_key = set(result_per_task['model'])

    if args.output_result:
        file_name=os.path.join('cache','result{}_per_task.csv'.format(args.output_prefix))
        result_list = []
        if os.path.exists(file_name):
            result_list.append(pd.read_csv(file_name,index_col=0))
        result_list.append(result_per_task)
        result_per_task_all = pd.concat(result_list, ignore_index=True)
        result_per_task_all.to_csv(file_name, header=True)



    key_individual_record=['test_max','max_prompt','max_abs_prompt'] if not (args.few_shot or args.augment) else ['test_max']

    result_table_dict={}
    for key in key_individual_record:
        file_name='result{}_'.format(args.output_prefix)+key+'_table.csv'
        file_name=os.path.join('cache', file_name)
        result_list=[]
        if os.path.exists(file_name):
            result_list.append(pd.read_csv(file_name,index_col='unique_task_id'))
        for model in models_key:
            table_per=result_per_task[result_per_task['model']==model][['unique_task_id',key]]
            table_per=table_per.set_index('unique_task_id')

            model_name=model if not args.augment else 'augment_'+args.augment_type+'_'+model

            table_per.columns=[model_name]
            result_list.append(table_per)
        result_table=pd.concat(result_list,axis=1)
        result_table=result_table.sort_index()

        if args.output_result:
            result_table.to_csv(file_name, header=True)
        result_table_dict[key] = result_table





    result_per_dataset=[]
    for model in models_key:
        result=result_per_task[result_per_task['model']==model]
        datasets_key=set(result['dataset'])
        for dataset in datasets_key:
            result_dataset=result[result['dataset']==dataset]
            mean_dataset=result_dataset[['train_mean','val_mean','test_mean','train_max','val_max','test_max']].mean()
            result_per_dataset.append([model,dataset,*mean_dataset])
    result_per_dataset=pd.DataFrame(result_per_dataset,columns=['model', 'dataset','train_mean','val_mean','test_mean','train_max','val_max','test_max'])





    datasets_key=list(datasets_key)
    subbenchmarks={'Average_bio':['hiv','bace','muv'],'Average_tox':['toxcast','tox21'],'Average_pha':['bbbp','cyp450'],'Average_bench':['hiv','bace','muv','toxcast','tox21','bbbp','cyp450'],'Average_phy':['esol','lipo','freesolv'],}

    result_per_dataset_table=result_per_dataset[['model','dataset','test_max']]
    result_per_dataset_table_permutated=[]
    for model in models_key:
        result_rec=[]
        subbenchmarks_result = {}
        for key in subbenchmarks.keys():
            subbenchmarks_result[key] = []
        for dataset in datasets_key:
            result=result_per_dataset_table.loc[(result_per_dataset_table['model']==model)&(result_per_dataset_table['dataset']==dataset),'test_max'].values[0]
            result_rec.append(result)
            for benchmark in subbenchmarks.keys():
                if dataset in subbenchmarks[benchmark]:
                    subbenchmarks_result[benchmark].append(result)
        subbenchmarks_result_list=[]
        for key in subbenchmarks.keys():
            subbenchmarks_result_list.append(np.mean(subbenchmarks_result[key]))
        result_per_dataset_table_permutated.append([model,args.task_type,args.augment_type,*result_rec,np.mean(result_rec),*subbenchmarks_result_list])

    result_per_dataset_table_permutated=pd.DataFrame(result_per_dataset_table_permutated,columns=['Method','task_type','augment_type',*datasets_key,'Average',*list(subbenchmarks.keys())])

    if args.output_result:
        result_per_dataset_table_permutated = result_per_dataset_table_permutated.reindex(sorted(result_per_dataset_table_permutated.columns), axis=1)
        # if args.few_shot:
        #     file_name='result_few_shot_result_per_dataset_table_permutated.csv'
        # # elif args.augment:
        # #     fime_name='result_augment_result_per_dataset_table_permutated.csv'
        # else:
        file_name=os.path.join('cache','result{}_result_per_dataset_table_permutated.csv'.format(args.output_prefix))
        result_list = []
        if os.path.exists(file_name):
            result_list.append(pd.read_csv(file_name,index_col=0))
        result_list.append(result_per_dataset_table_permutated)
        result_per_dataset_table_permutated_all = pd.concat(result_list,ignore_index=True)
        result_per_dataset_table_permutated_all.to_csv(file_name, header=True)



    average_over_tasks=[]
    for model in models_key:
        average_over_tasks.append([model,args.task_type,args.augment_type,float(result_per_task.loc[result_per_task['model']==model,'test_max'].mean())])

    average_over_tasks=pd.DataFrame(average_over_tasks,columns=['Model','task_type','augment_type','Average'])

    if args.output_result:
        file_name = 'result{}_average_over_tasks.csv'.format(args.output_prefix)
        # file_name = 'result_few_shot_average_over_tasks.csv' if args.few_shot else 'result_average_over_tasks.csv'
        result_list = []
        if os.path.exists(file_name):
            result_list.append(pd.read_csv(file_name,index_col=0))
        result_list.append(average_over_tasks)
        average_over_tasks_all = pd.concat(result_list,ignore_index=True)
        average_over_tasks_all.to_csv(file_name, header=True)


