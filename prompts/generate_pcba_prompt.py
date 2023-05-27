import pandas as pd
import numpy as np
import os
import os
import commentjson
import json
import openai

with open('../openai_api_key.json') as f:
    key = json.load(f)
    openai.api_key=key

def askchatgpt(question):

    completion = openai.Completion.create(model="gpt-3.5-turbo", \
                                          messages=[{"role": "user", "content": question}], \
                                          api_base="https://api.openai.com/v1/chat")

    return completion['choices'][0]['message']['content'].replace("\n", "")


with open(os.path.join('..','prompts','downstream_task_prompt_multitask_new.json'),'r',encoding='utf8') as f:
    prompt_file=commentjson.load(f)

data_assay=[]
with open(os.path.join('..','prompt_data','pcba_descriptions.txt'),'r') as f:
    while True:
        line = f.readline()
        if not line: # EOF
            break
        context = commentjson.loads(line)
        data_assay.append([context['id'],context['description']])
data_assay=pd.DataFrame(data_assay,columns=['id', 'description'])


data_pcba=pd.read_csv(os.path.join('..','property_data','pcba','raw','pcba.csv'))


meta_prompt='Please remove paragraphs which is paper references, and output the left unchanged:\n{}'
meta_prompt_2='summarize the assay: \n{}'

masking_words=['1Description',]


for id,assay_id in enumerate(data_pcba.columns):
    print('id: {}'.format(id))
    if str(id) in prompt_file['pcba']:
        continue
    if 'PCBA' in assay_id:
        prompt = ''
        description=data_assay[(data_assay['id']==assay_id.replace('PCBA-',''))]['description'].values[0]
        print('Original:\n {}\n'.format(description))
        if 'References:' in description:
            description=description[0:description.find('References')]
        paragraph = description.split('\n')
        paragraph_new=[]
        for para in paragraph:
            if len(para)>150:
                paragraph_new.append(para)
        description_new=''
        for para in paragraph_new:
            description_new+=para
            description_new += '\n'
        print('Formatted:\n {}\n'.format(description_new))
        description_new=askchatgpt(meta_prompt.format(description_new))
        print('No reference:\n {}\n'.format(description_new))
        description_new=askchatgpt(meta_prompt_2.format(description_new))
        print('No reference:\n {}\n'.format(description_new))
        prompt+=description_new
        prompt += ' ' + 'Is the molecule effective to this assay?'
        prompt_file['pcba'][str(id)]=[prompt]
        with open(os.path.join('..','prompts','downstream_task_prompt_multitask_new.json'),'w') as f:
            json.dump(prompt_file,f,indent=2)


# data_assay_pd.loc(task_assays_name[256])

# des_columns=[]
# for key in data_assay_pd.keys():
#     if len(str(data_assay_pd[key][0]))>100:
#         des_columns.append(key)
#
# des_columns=['assay_source_desc','assay_desc','assay_component_desc','assay_component_target_desc','assay_component_endpoint_desc']
# data_assay_pd.set_index("assay_component_endpoint_name",inplace=True)
# # assay_name_in=[]
# filterd_data={}
# # label_id=[]
# for task_id,assay_name in enumerate(task_assays_name):
#     if assay_name in data_assay_pd['assay_name']:
#         # label_id.append(task_id)
#         filterd_data[task_id]=(data_assay_pd.loc[assay_name])
#
# # filtered_data_assay_pd=data_assay_pd.loc[assay_name_in,des_columns]
#
# prompt_toxcast={}
# for task_id in filterd_data.keys():
#     assay_info=filterd_data[task_id]
#     prompt=''
#     prompt+=assay_info['assay_component_desc']
#     prompt+=assay_info['assay_component_endpoint_desc']
#     prompt+=' '+'Is this molecule effective to this assay?'
#     prompt_toxcast[task_id] = [prompt]
#
# prompt_file['toxcast']=prompt_toxcast
# with open(os.path.join('..','downstream_task_prompt_multitask_new.json'),'w') as f:
#     prompt_file=commentjson.dump(prompt_file,f)
# print(1)



