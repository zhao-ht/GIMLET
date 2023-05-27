import openai
import argparse
import json
import commentjson
import os
from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument("--prompt_file",type=str,default=None)

args = parser.parse_args()



with open('../openai_api_key.json') as f:
    key = json.load(f)
    openai.api_key=key

def askchatgpt(question):

    completion = openai.Completion.create(model="gpt-3.5-turbo", \
                                          messages=[{"role": "system", "content" : "You are required to rewrite the text if asked. I won't provide you any additional information, and do not ask question even if the text required you to rewrite is not feasble."},{"role": "user", "content": question}], \
                                          api_base="https://api.openai.com/v1/chat")

    return completion['choices'][0]['message']['content'].replace("\n", "")



with open(args.prompt_file,encoding='utf-8') as f:
    prompt_file = commentjson.load(f)



if os.path.exists('augmented_'+args.prompt_file):
    with open('augmented_'+args.prompt_file, encoding='utf-8') as f:
        prompt_augmented = commentjson.load(f)
else:
    print("Warning: Creating a new augmented prompt file")
    prompt_augmented ={}



generate_prompt_prompts={'rewrited':'Rephrase the text  of the following prompt; do not ask additional questions: \n',
                         'expanded':'Rephrase the text  of the following prompt longer; do not ask additional questions: \n',
                         'detailed':'Rephrase the text  of the following prompt by adding more explanation; do not ask additional questions:  \n',
                         'shortened':'Rephrase the text  of the following prompt shorter: \n'}
generate_question_prompts={'rewrited':'Rephrase the text  of the following question; do not ask additional questions: \n',
                         'expanded':'Rephrase the text  of the following question longer; do not ask additional questions: \n',
                         'detailed':'Rephrase the text  of the following question; do not ask additional questions:  \n',
                         'shortened':'Rephrase the text  of the following question shorter: \n'}


generate_types=['rewrited','expanded','detailed','shortened']
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
        prompt_new = askchatgpt(generate_prompt_prompts[type] + prompt)
        if question is not None:
            prompt_new += askchatgpt(generate_question_prompts[type] + question)
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
        with open('augmented_'+args.prompt_file,'w') as f:
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






