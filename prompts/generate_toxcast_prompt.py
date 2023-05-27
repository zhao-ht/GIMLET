import pandas as pd
import numpy as np
import os
import os
import commentjson

# for file in os.listdir('ToxCast_Assay_Information'):
#     if file.startswith("~") and file.endswith(".csv"):
#         print(file)

with open(os.path.join('..','downstream_task_prompt_multitask.json'),'r') as f:
    prompt_file=commentjson.load(f)

data_assay_pd=pd.read_excel(os.path.join('ToxCast_Assay_Information',
                                   'Assay_Summary_210920.xlsx'))

data_toxcast=pd.read_csv(os.path.join('../','property_data','toxcast','raw','toxcast_data.csv'))

task_assays_name=data_toxcast.keys()[1:]


# data_assay_pd.loc(task_assays_name[256])

des_columns=[]
for key in data_assay_pd.keys():
    if len(str(data_assay_pd[key][0]))>100:
        des_columns.append(key)

des_columns=['assay_source_desc','assay_desc','assay_component_desc','assay_component_target_desc','assay_component_endpoint_desc']
data_assay_pd.set_index("assay_component_endpoint_name",inplace=True)
# assay_name_in=[]
filterd_data={}
# label_id=[]
for task_id,assay_name in enumerate(task_assays_name):
    if assay_name in data_assay_pd['assay_name']:
        # label_id.append(task_id)
        filterd_data[task_id]=(data_assay_pd.loc[assay_name])

# filtered_data_assay_pd=data_assay_pd.loc[assay_name_in,des_columns]

prompt_toxcast={}
for task_id in filterd_data.keys():
    assay_info=filterd_data[task_id]
    prompt=''
    prompt+=assay_info['assay_component_desc']
    prompt+=assay_info['assay_component_endpoint_desc']
    prompt+=' '+'Is this molecule effective to this assay?'
    prompt_toxcast[task_id] = [prompt]

prompt_file['toxcast']=prompt_toxcast
with open(os.path.join('..','downstream_task_prompt_multitask_new.json'),'w') as f:
    prompt_file=commentjson.dump(prompt_file,f)
print(1)



