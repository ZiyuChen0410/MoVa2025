from openai import OpenAI
import os
import time
import datetime
import pytz
import configparser
import regex
import pandas as pd
from anytree import Node, RenderTree
import tiktoken
from itertools import islice
import requests
from tenacity import retry, wait_random_exponential, stop_after_attempt
from openai import OpenAI
import anthropic
import json
import hashlib
import pickle
from typing import Optional, Any
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from llama_cpp import Llama
from sklearn import metrics
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple
from tenacity import retry, wait_random_exponential, stop_after_attempt, retry_if_exception_type


CACHE_DIR = "data/api_call_cache"

def _cache_key(prompt: str) -> str:
    return hashlib.sha1(prompt.encode()).hexdigest()

def _cache_file_path(prompt: str, cache_dir:str, api_type: str, deployment_name:str) -> str:
    cache_key = _cache_key(prompt)
    api_dir = os.path.join(cache_dir, api_type)
    model_dir = os.path.join(api_dir, deployment_name)
    os.makedirs(model_dir, exist_ok=True)
    return os.path.join(model_dir, f"{cache_key}.pkl")

def _load_from_cache(prompt: str, cache_dir:str, api_type: str, deployment_name:str) -> Optional[Any]:
    cache_file = _cache_file_path(prompt, cache_dir, api_type, deployment_name)
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            return pickle.load(f)
    return None

def _save_to_cache(prompt: str, response: Any, cache_dir:str, api_type: str, deployment_name:str) -> None:
    cache_file = _cache_file_path(prompt, cache_dir, api_type, deployment_name)
    with open(cache_file, "wb") as f:
        pickle.dump(response, f)

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(30))
def api_call(api,deployment_name,api_key, prompt,  temperature, max_tokens, top_p, logprobs, top_logprobs, overwrite = False):
    """
    Call API (OpenAI, Azure, Perplexity) and return response
    - prompt: prompt template
    - deployment_name: name of the deployment to use (e.g. gpt-4, gpt-3.5-turbo, etc.)
    - temperature: temperature parameter
    - max_tokens: max tokens parameter
    - top_p: top p parameter
    """
    try:
        time.sleep(1)  # Change to avoid rate limit:0.5,1,2

        if overwrite:
            cached_response = None
        else:
            cached_response = _load_from_cache(prompt, CACHE_DIR, deployment_name.split('-', 1)[0], deployment_name)
        if cached_response is not None:
            return cached_response

        if deployment_name in ["deepseek-chat","deepseek-reasoner","gpt-35-turbo", "gpt-4", "gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o", "gpt-4-turbo"]:
            if api=='openai':
                client = OpenAI(api_key=api_key)
            elif api=='deepseek':
                client =  OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

            response = client.chat.completions.create(
                model=deployment_name,
                temperature=float(temperature),
                max_tokens=int(max_tokens),
                top_p=float(top_p),
                logprobs = logprobs,
                top_logprobs = top_logprobs,
                messages=[
                    {"role": "system", "content": ""},
                    {"role": "user", "content": prompt},
                ],
            )
            _save_to_cache(prompt, response.choices[0],
                        CACHE_DIR, deployment_name.split('-', 1)[0], deployment_name)
            return response.choices[0]

        elif deployment_name in [
            "llama-2-70b-chat",
            "codellama-34b-instruct",
            # "mistral-7b-instruct",
        ]:
            payload = {
                "model": deployment_name,
                "temperature": float(temperature),
                "max_tokens": int(max_tokens),
                "top_p": float(top_p),
                "request_timeout": 1000,
                "messages": [
                    {"role": "system", "content": ""},
                    {"role": "user", "content": prompt},
                ],
            }
            headers = {
                "accept": "application/json",
                "content-type": "application/json",
                "authorization": PERPLEXITY_API_KEY,
            }
            response = requests.post(
                "https://api.perplexity.ai/chat/completions", json=payload, headers=headers
            )
            if response.status_code != 200:
                print(response.status_code)
                print(response.text)
                raise Exception("Error in perplexity API call")
            _save_to_cache(prompt, response.json()["choices"][0]["message"]["content"],
                        CACHE_DIR, deployment_name.split('-', 1)[0], deployment_name)
            return response.json()["choices"][0]["message"]["content"]
        elif deployment_name in [
            "gemma-2-9b-it",
            "gemma-2-27b-it",
        ]:
            pipe = pipeline(
                "text-generation",
                model=f"google/{deployment_name}",
                model_kwargs={"torch_dtype": torch.bfloat16},
                device="cuda",
            )

            messages = [
                {"role": "user", "content": prompt},
            ]
            with torch.no_grad():
                outputs = pipe(
                    messages,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    batch_size=1
                )
            response = outputs[0]["generated_text"][-1]["content"]
            return response
        elif deployment_name in [
            "gemma-2-9b-it-GGUF"
        ]:
            messages = [ 
                {"role": "user", "content": prompt}]  

            llm = Llama.from_pretrained(
                repo_id=f"bartowski/{deployment_name}",
                filename="*Q4_K_Lgguf",
                verbose=False,
                chat_format="gemma",
            )

            x = llm.create_chat_completion(
                messages=messages,
            )

            response = (x["choices"][0]["message"]["content"])
            return response
        elif deployment_name in [
            "Phi-3-mini-128k-instruct",
            "Phi-3-mini-4k-instruct"
        ]:
            model_dir = f"microsoft/{deployment_name}"
            model = AutoModelForCausalLM.from_pretrained(
            model_dir, 
            device_map="cuda", 
            torch_dtype=torch.float16, 
            trust_remote_code=True, 
            )

            tokenizer = AutoTokenizer.from_pretrained(model_dir)
            messages = [
                {"role": "user", "content": prompt},
            ]
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                batch_size=1
            )
            generation_args = {
                "max_new_tokens": max_tokens,
                "return_full_text": False,
                "temperature": 0.0,
                "do_sample": False,
            }
            output = pipe(messages, **generation_args)
            response = output[0]['generated_text']
            _save_to_cache(prompt, response,
                        CACHE_DIR, deployment_name.split('-', 1)[0], deployment_name)
            return response
        elif deployment_name in [
            "mistral-7b-instruct",
        ]:
            model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", torch_dtype=torch.float16)
            tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

            messages = [
                {"role": "user", "content": f"{prompt}"}
            ]

            encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")
            model_inputs = encodeds.to("cuda")
            model.to("cuda")

            generated_ids = model.generate(model_inputs, max_new_tokens=max_tokens, do_sample=True)
            decoded = tokenizer.batch_decode(generated_ids)
            response = decoded[0]
            return response
        elif deployment_name in [
            "claude-v1",
            "claude-v1.2",
            "claude-v1.3",
            "claude-v1.4",
            "claude-2-instant",
            "claude-2",
            "claude-3-haiku-20240307",
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229"
        ]:
            client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
            if deployment_name in [
                "claude-3-opus-20240229",
                "claude-3-haiku-20240307",
                "claude-3-sonnet-20240229"
            ]:
                response = client.messages.create(
                    model=deployment_name,
                    max_tokens=max_tokens,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                response = json.loads(response.model_dump_json())["content"][0]["text"]
            else:
                completion = client.completions.create(
                    prompt=f"{anthropic.HUMAN_PROMPT} {prompt}{anthropic.AI_PROMPT}",
                    stop_sequences=[anthropic.HUMAN_PROMPT],
                    model=deployment_name,
                    max_tokens_to_sample=max_tokens,
                )
                response = completion.completion
            _save_to_cache(prompt, response, CACHE_DIR, deployment_name.split('-', 1)[0], deployment_name)
            return response
        else:
            print("Invalid deployment name. Please try again.")
    except Exception as e:
        print(f"Error encountered: {e}. Sleeping for 10 seconds before retry.")
        time.sleep(10)
        raise e

