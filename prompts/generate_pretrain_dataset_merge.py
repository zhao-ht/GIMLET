import pandas as pd
import os
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--prompt_dir', type=str, default='../pretrain_datasets')
parser.add_argument('--merge_file_list', type=str, nargs='+',)
parser.add_argument('--merge_file_policy', type=str, default='makeup')
parser.add_argument('--merge_file_ratio', type=float, nargs='+',default=[])
parser.add_argument('--final_file_name',type=str,default=None)
parser.add_argument('--split_part',type=int,default=1)


args = parser.parse_args()



# all the csv should have the same columns,
df_list=[]
df_len_list=[]
for file in args.merge_file_list:
    df=pd.read_csv(os.path.join(args.prompt_dir,file))
    df_len_list.append(len(df))
    print(file)
    print(df.columns)
    print(len(df))
    df_list.append(df)

if args.merge_file_policy=='custom':
    df_ratio_list=args.merge_file_ratio
elif args.merge_file_policy=='makeup':
    max_len=max(df_len_list)
    df_ratio_list=[max_len/length for length in df_len_list]
else:
    raise ValueError("Not supported merge_file_policy yet")

if args.final_file_name is None:
    args.final_file_name='merge'
    for ind,name in enumerate(args.merge_file_list):
        args.final_file_name+='_'+name.replace('.csv','')+'_'+str(round(df_ratio_list[ind],2))
    args.final_file_name+='.csv'
    print('saving to ',args.final_file_name)


df_to_merge=[]
for df_ratio,df in zip(df_ratio_list,df_list):
    if df_ratio==1:
        df_to_merge.append(df)
        print(df.columns)
        print(len(df))
    else:
        while df_ratio>1:
            df_to_merge.append(df)
            print(df.columns)
            print(len(df))
            df_ratio-=1
        df=df.sample(frac=df_ratio,replace=False)
        df_to_merge.append(df)
        print(df.columns)
        print(len(df))
result = pd.concat(df_to_merge, ignore_index=True, sort=False)
if 'assayid' in result:
    result['assayid'] = result['assayid'].fillna('')
print(len(result))
# result = pd.read_csv(args.final_file_name)
if args.split_part>1:
    sample_num=int(1.0/args.split_part*len(result))
    result_left=result
    for i in range(args.split_part-1):
        result_part = result_left.sample(n=sample_num,replace=False)
        result_left = result_left[~result_left.index.isin(result_part.index)]
        print(len(result_part))
        print(len(result_left))
        result_part.to_csv(os.path.join(args.prompt_dir,'split_'+str(i)+'_'+args.final_file_name))
    result_left.to_csv(os.path.join(args.prompt_dir,'split_'+str(args.split_part-1)+'_'+args.final_file_name))

    # result.to_csv(args.final_file_name)
else:
    result.to_csv(os.path.join(args.prompt_dir,args.final_file_name))
