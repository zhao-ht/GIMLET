
import torch  
import openai  
import os
import time

class LLM(torch.nn.Module):  
    def __init__(  
        self,  
        model="text-davinci-003",  
        stop=["\n"],  
        temperature=0,  
        max_tokens=100,  
        top_p=1,  
        frequency_penalty=0.0,  
        presence_penalty=0.0,
    ):  
        super(LLM, self).__init__()  

        openai.api_key = os.environ["OPENAI_API_KEY"]
        self.model = model  
        self.stop = stop  
        self.temperature = temperature  
        self.max_tokens = max_tokens  
        self.top_p = top_p  
        self.frequency_penalty = frequency_penalty  
        self.presence_penalty = presence_penalty  
        self.max_retries = 10
        self.retry_interval = 60 
        
    def generate(self, prompt):  
        retries = 0  
        while retries < self.max_retries:  
            try:  
                response = openai.Completion.create(  
                    model=self.model,  
                    prompt=prompt,  
                    temperature=self.temperature,  
                    max_tokens=self.max_tokens,  
                    top_p=self.top_p,  
                    frequency_penalty=self.frequency_penalty,  
                    presence_penalty=self.presence_penalty, 
                    logprobs=100,
                    logit_bias={2949: 10, 5297: 10, 198:5}, # bias against yes, no, and stop token
                )  
                return response
            except:  
                retries += 1  
                print(f"Rate limit exceeded. Retrying in {self.retry_interval} seconds... (Retry {retries}/{self.max_retries})")  
                time.sleep(self.retry_interval)  
        
        