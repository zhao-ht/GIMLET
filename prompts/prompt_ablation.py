import openai
import argparse
import json
import commentjson
import os
from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument("--prompt_file",type=str,default=None)

args = parser.parse_args()

file_prefix='ablated_'

with open('../openai_api_key.json') as f:
    key = json.load(f)
    openai.api_key=key

def askchatgpt(question):

    completion = openai.Completion.create(model="gpt-3.5-turbo", \
                                          messages=[{"role": "user", "content": question}], \
                                          api_base="https://api.openai.com/v1/chat")

    return completion['choices'][0]['message']['content'].replace("\n", "")



with open(args.prompt_file,encoding='utf-8') as f:
    prompt_file = commentjson.load(f)



if os.path.exists(file_prefix+args.prompt_file):
    with open(file_prefix+args.prompt_file, encoding='utf-8') as f:
        prompt_augmented = commentjson.load(f)
else:
    print("Warning: Creating a new ablated_ prompt file")
    prompt_augmented ={}



generate_prompt_prompts={'name':'Here is an assay: {}. \n Print only the target name of this assay with no additional words.'}
generate_prompt_templete={'name':'The assay name is {}.'}

generate_types=['name']
ignored_keys=['pcba']

def augment_by_chatgpt(prompt,result_for_key):
    prompt_sentence = prompt.split('.')
    if '?' in prompt_sentence[-1] and len(prompt_sentence) > 1:
        question = prompt_sentence.pop(-1)
    else:
        question = None
    prompt = ''
    for sentence in prompt_sentence:
        prompt += sentence + ', '
    prompt = prompt[0:-2] + '.'

    print('original: ', prompt)
    print('original question: ',question)
    for type in generate_types:
        prompt_new = askchatgpt(generate_prompt_prompts[type].format(prompt.replace('\n','')))
        prompt_new=generate_prompt_templete[type].format(prompt_new)
        prompt_new=prompt_new.replace('..','.')
        if question is not None:
            prompt_new +=question
        print(type, ': ', prompt_new)
        result_for_key[type].append(prompt_new)

    print('\n')
    return result_for_key

def recurrent_augment_prompts_file(prompt_file_part,prompt_augmented_part):
    if isinstance(prompt_file_part,list):
        for type in generate_types:
            prompt_augmented_part[type]=[]
        for prompt in prompt_file_part:
            prompt_augmented_part=augment_by_chatgpt(prompt,prompt_augmented_part)
        with open(file_prefix+args.prompt_file,'w') as f:
            json.dump(prompt_augmented,f,indent=2)
    else:
        assert isinstance(prompt_file_part,dict)
        for key in prompt_file_part:
            if key in ignored_keys:
                continue
            if isinstance(prompt_file_part[key],dict) or key not in prompt_augmented_part:
                print(key)
                if key not in prompt_augmented_part:
                    prompt_augmented_part[key]=dict()
                recurrent_augment_prompts_file(prompt_file_part[key],prompt_augmented_part[key])



recurrent_augment_prompts_file(prompt_file,prompt_augmented)

# for key in tqdm(prompt_file.keys()):
#     if key not in prompt_augmented:
#         result_for_key={}
#         for type in generate_types:
#             result_for_key[type]=[]
#         for prompt in prompt_file[key]:
#             augment_by_chatgpt(prompt,result_for_key)
#         prompt_augmented[key]=result_for_key
#         with open('augmented_'+args.prompt_file,'w') as f:
#             json.dump(prompt_augmented,f,indent=2)






