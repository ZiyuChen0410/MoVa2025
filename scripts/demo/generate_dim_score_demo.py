import argparse
import regex
import gzip
import json
import re
import math
import pandas as pd
import concurrent.futures
import time
from typing import Optional
from scripts.demo import data_process, api_call
import copy

def extract_json_content(response_content):
    """
    Extract JSON-like content from a response string.
    """
    json_pattern = r'(?:```json\s*)?(\{.*?\})(?:\s*```)?'  # Matches the first JSON-like object in the response
    match = re.search(json_pattern, response_content, re.DOTALL)
    if match:
        return match.group(0)
    
# --- MAC version of get_dims ---
def get_dims_mac(
    api,
    api_key,
    prompt, 
    moral_stat,
    deployment_name,
    temperature,
    max_tokens,
    top_p,
    logprobs,
    top_logprobs,

):
    generation_prompt = prompt.format(text=moral_stat[1])

    response = api_call.api_call(
                api, deployment_name, api_key, generation_prompt, temperature, max_tokens, top_p, logprobs, top_logprobs
    )

    print("MAC response message:", response.message.content)
    print("end of response")

    dim_dict = {"family": None, "group": None, "reciprocity": None, "heroism": None, "deference": None,
                "fairness": None, "property": None}
    dim_answer_dict = {"family": None, "group": None, "reciprocity": None, "heroism": None, "deference": None,
                       "fairness": None, "property": None}
    
    logprobs_items = response.logprobs.content if response.logprobs else []
    if response.message.content:#regex.match(target_format, response.message.content):
        token_prob_list = []
        dim_answer_list = []
        for logprob in logprobs_items:
            if logprob.token in ["0", "1"]:
                anti_token = "1" if logprob.token == "0" else "0"
                tokens = [top.token for top in logprob.top_logprobs]
                if anti_token in tokens:
                    curr_poss = math.exp(logprob.logprob) / (math.exp(logprob.top_logprobs[tokens.index(anti_token)].logprob) + math.exp(logprob.logprob))
                else:
                    curr_poss = 1
                token_prob_list.append((logprob.token, curr_poss))
                dim_answer_list.append(logprob.token)
        confidence_score = []
        for token, prob in token_prob_list:
            confidence_score.append(1 - prob if token=='0' else prob)
        dim_dict["family"] = confidence_score[0]
        dim_dict["group"] = confidence_score[1]
        dim_dict["reciprocity"] = confidence_score[2]
        dim_dict["heroism"] = confidence_score[3]
        dim_dict["deference"] = confidence_score[4]
        dim_dict["fairness"] = confidence_score[5]
        dim_dict["property"] = confidence_score[6]

        dim_answer_dict["family"] = dim_answer_list[0]
        dim_answer_dict["group"] = dim_answer_list[1]
        dim_answer_dict["reciprocity"] = dim_answer_list[2]
        dim_answer_dict["heroism"] = dim_answer_list[3]
        dim_answer_dict["deference"] = dim_answer_list[4]
        dim_answer_dict["fairness"] = dim_answer_list[5]
        dim_answer_dict["property"] = dim_answer_list[6]

    print("MAC dim_dict", dim_dict)

    return response, dim_dict, dim_answer_dict

# --- HumVal version of get_dims  ---
def get_dims_humval(
    theory,
    api,
    api_key,
    prompt, 
    moral_stat,
    deployment_name,
    temperature,
    max_tokens,
    top_p,
    logprobs,
    top_logprobs,

):

    generation_prompt = prompt.format(text=moral_stat[1])
    print("HumVal Prompt:", generation_prompt)

    response = api_call.api_call(
                api, deployment_name, api_key, generation_prompt, temperature, max_tokens, top_p, logprobs, top_logprobs
    )
    
    print("HumVal response message:", response.message.content)
    print("end of response")
    if theory in ['value10']:
        dim_dict =  {"power": None, "achievement": None, "hedonism": None, "stimulation": None, "self-direction": None,
                     "universalism": None, "benevolence": None, "tradition": None, "conformity": None, "security": None}
        dim_answer_dict =  {"power": None, "achievement": None, "hedonism": None, "stimulation": None, "self-direction": None,
                            "universalism": None, "benevolence": None, "tradition": None, "conformity": None, "security": None}
    elif theory=='value20':
        dim_dict = {"power - dominance": None, "power - resources": None, "power - face": None, "achievement": None,
                    "hedonism": None, "stimulation": None, "self-direction - thought": None, "self-direction - action": None,
                    "universalism - concern": None, "universalism - nature": None, "universalism - tolerance": None,
                    "universalism - objectivity": None, "benevolence - caring": None, "benevolence - dependability": None,
                    "tradition": None, "humility": None, "conformity - rules": None, "conformity - interpersonal": None,
                    "security - personal": None, "security - societal": None}
        dim_answer_dict = copy.deepcopy(dim_dict)
    

    # For theories handled here we use the logprobs values
    logprobs_items = response.logprobs.content if response.logprobs else []
    if theory in ['value10','value20']:#insert direction here, adjust U,Y,N 
        if response.logprobs:
            token_prob_list = []
            dim_answer_list = []
            for logprob in logprobs_items:
                if logprob.token in ["0", "1"]:
                    anti_token = "1" if logprob.token=="0" else "0"
                    tokens = [top.token for top in logprob.top_logprobs]
                    if anti_token in tokens:
                        curr_poss = math.exp(logprob.logprob) / (math.exp(logprob.top_logprobs[tokens.index(anti_token)].logprob)+ math.exp(logprob.logprob))
                    else:
                        curr_poss = 1
                    token_prob_list.append((logprob.token, curr_poss))
                    dim_answer_list.append(logprob.token)
            confidence_score = [1 - prob if token=='0' else prob for token, prob in token_prob_list]
            if theory in ['value10']:
                dim_dict["power"] = confidence_score[0]
                dim_dict["achievement"] = confidence_score[1]
                dim_dict["hedonism"] = confidence_score[2]
                dim_dict["stimulation"] = confidence_score[3]
                dim_dict["self-direction"] = confidence_score[4]
                dim_dict["universalism"] = confidence_score[5]
                dim_dict["benevolence"] = confidence_score[6]
                dim_dict["tradition"] = confidence_score[7]
                dim_dict["conformity"] = confidence_score[8]
                dim_dict["security"] = confidence_score[9]
                dim_answer_dict["power"] = dim_answer_list[0]
                dim_answer_dict["achievement"] = dim_answer_list[1]
                dim_answer_dict["hedonism"] = dim_answer_list[2]
                dim_answer_dict["stimulation"] = dim_answer_list[3]
                dim_answer_dict["self-direction"] = dim_answer_list[4]
                dim_answer_dict["universalism"] = dim_answer_list[5]
                dim_answer_dict["benevolence"] = dim_answer_list[6]
                dim_answer_dict["tradition"] = dim_answer_list[7]
                dim_answer_dict["conformity"] = dim_answer_list[8]
                dim_answer_dict["security"] = dim_answer_list[9]
            elif theory=='value20':
                dim_dict["power - dominance"] = confidence_score[0]
                dim_dict["power - resources"] = confidence_score[1]
                dim_dict["power - face"] = confidence_score[2]
                dim_dict["achievement"] = confidence_score[3]
                dim_dict["hedonism"] = confidence_score[4]
                dim_dict["stimulation"] = confidence_score[5]
                dim_dict["self-direction - thought"] = confidence_score[6]
                dim_dict["self-direction - action"] = confidence_score[7]
                dim_dict["universalism - concern"] = confidence_score[8]
                dim_dict["universalism - nature"] = confidence_score[9]
                dim_dict["universalism - tolerance"] = confidence_score[10]
                dim_dict["universalism - objectivity"] = confidence_score[11]
                dim_dict["benevolence - caring"] = confidence_score[12]
                dim_dict["benevolence - dependability"] = confidence_score[13]
                dim_dict["tradition"] = confidence_score[14]
                dim_dict["humility"] = confidence_score[15]
                dim_dict["conformity - rules"] = confidence_score[16]
                dim_dict["conformity - interpersonal"] = confidence_score[17]
                dim_dict["security - personal"] = confidence_score[18]
                dim_dict["security - societal"] = confidence_score[19]

                dim_answer_dict["power - dominance"] = dim_answer_list[0]
                dim_answer_dict["power - resources"] = dim_answer_list[1]
                dim_answer_dict["power - face"] = dim_answer_list[2]
                dim_answer_dict["achievement"] = dim_answer_list[3]
                dim_answer_dict["hedonism"] = dim_answer_list[4]
                dim_answer_dict["stimulation"] = dim_answer_list[5]
                dim_answer_dict["self-direction - thought"] = dim_answer_list[6]
                dim_answer_dict["self-direction - action"] = dim_answer_list[7]
                dim_answer_dict["universalism - concern"] = dim_answer_list[8]
                dim_answer_dict["universalism - nature"] = dim_answer_list[9]
                dim_answer_dict["universalism - tolerance"] = dim_answer_list[10]
                dim_answer_dict["universalism - objectivity"] = dim_answer_list[11]
                dim_answer_dict["benevolence - caring"] = dim_answer_list[12]
                dim_answer_dict["benevolence - dependability"] = dim_answer_list[13]
                dim_answer_dict["tradition"] = dim_answer_list[14]
                dim_answer_dict["humility"] = dim_answer_list[15]
                dim_answer_dict["conformity - rules"] = dim_answer_list[16]
                dim_answer_dict["conformity - interpersonal"] = dim_answer_list[17]
                dim_answer_dict["security - personal"] = dim_answer_list[18]
                dim_answer_dict["security - societal"] = dim_answer_list[19]

    print("HumVal dim_dict", dim_dict)
    print("HumVal dim_answer_dict", dim_answer_dict)
    return response, dim_dict, dim_answer_dict

# --- MFT version of get_dims ---
def get_dims_mft(
    api,
    api_key,
    prompt, 
    moral_stat,
    deployment_name,
    temperature,
    max_tokens,
    top_p,
    logprobs,
    top_logprobs,
    with_liberty,
):
    generation_prompt = prompt.format(text=moral_stat[1])
    print("Prompt:", generation_prompt)
    response = api_call.api_call(
                api, deployment_name, api_key, generation_prompt, temperature, max_tokens, top_p, logprobs, top_logprobs
    )
    print(response.message.content)
    print("end of response")
    if with_liberty:
        dim_dict = {"care": None, "fairness": None, "loyalty": None, "authority": None, "sanctity": None, "liberty": None}
        dim_answer_dict = {"care": None, "fairness": None, "loyalty": None, "authority": None, "sanctity": None, "liberty": None}
    else:
        dim_dict = {"care": None, "fairness": None, "loyalty": None, "authority": None, "sanctity": None}
        dim_answer_dict = {"care": None, "fairness": None, "loyalty": None, "authority": None, "sanctity": None}
 
    if response.message.content:
        logprobs_items = response.logprobs.content
        token_prob_list = []
        dim_answer_list = []
        for logprob in logprobs_items:
            if logprob.token in ["0", "1"]:
                anti_token = "1" if logprob.token=="0" else "0"
                tokens = [top.token for top in logprob.top_logprobs]
                if anti_token in tokens:
                    curr_poss = math.exp(logprob.logprob) / (math.exp(logprob.top_logprobs[tokens.index(anti_token)].logprob) + math.exp(logprob.logprob))
                else:
                    curr_poss = 1
                token_prob_list.append((logprob.token, curr_poss))
                dim_answer_list.append(logprob.token)
        print(token_prob_list)
        confidence_score = []
        for token, prob in token_prob_list:
            confidence_score.append(1 - prob if token=='0' else prob)
        dim_dict["care"] = confidence_score[0]
        dim_dict["fairness"] = confidence_score[1]
        dim_dict["loyalty"] = confidence_score[2]
        dim_dict["authority"] = confidence_score[3]
        dim_dict["sanctity"] = confidence_score[4]
        dim_answer_dict["care"] = dim_answer_list[0]
        dim_answer_dict["fairness"] = dim_answer_list[1]
        dim_answer_dict["loyalty"] = dim_answer_list[2]
        dim_answer_dict["authority"] = dim_answer_list[3]
        dim_answer_dict["sanctity"] = dim_answer_list[4]
        if with_liberty:
            dim_dict["liberty"] = confidence_score[5]
            dim_answer_dict["liberty"] = dim_answer_list[5] 

    print(dim_dict)
    return response, dim_dict, dim_answer_dict

def get_dims_common_morality(
    api,
    api_key,
    prompt, 
    moral_stat,
    deployment_name,
    temperature,
    max_tokens,
    top_p,
    logprobs,
    top_logprobs,
    mode,
):
    
    if mode in ["common_morality"]:
        generation_prompt = prompt.format(text=moral_stat[1])
    print("Prompt:", generation_prompt)
    # Call the API with the generated prompt
    response = api_call.api_call(
                api, deployment_name, api_key, generation_prompt, temperature, max_tokens, top_p, logprobs, top_logprobs
    )

    response_content = response.message.content

    print(response.message.content)
    print("end of reponse")

    if mode in ["common_morality"]:
        dim_dict = {"death": [],
                         "pain": [],
                         "disable":[],
                         "freedom": [],
                         "pleasure": [],
                         "deceive": [],
                         "cheat": [],
                         "break_promise": [],
                         "break_law": [],
                         "duty": []}
        dim_answer_dict = copy.deepcopy(dim_dict)

    # Extract JSON-like content from the response
    response_content = extract_json_content(response_content)


    if response_content:
        # Extract logprobs for 0 and 1
        logprobs_items = response.logprobs.content

        # Filter for 0 and 1 tokens
        token_prob_list = []
        dim_answer_list = []
        counter = 0
        for logprob in logprobs_items:
            if logprob.token in ["0", "1"]:
                counter += 1
                if logprob.token == "0":
                    anti_token = "1"
                elif logprob.token == "1":
                    anti_token = "0"
                
                tokens = [top.token for top in logprob.top_logprobs]
                if anti_token in tokens:
                    # print(f"tokens for {list(dim_answer_dict.keys())[counter - 1]}: {tokens}")
                    # the anti token is in the top tokens, then calculate the probability with scaling such that 0 and 1 take up the whole thing
                    curr_poss = math.exp(logprob.logprob) / (math.exp(logprob.top_logprobs[tokens.index(anti_token)].logprob) + math.exp(logprob.logprob))
                else:
                    # anti token not there, bump the probability to one
                    curr_poss = 1
                token_prob_list.append((logprob.token, curr_poss))
                dim_answer_list.append(logprob.token)

        confidence_score = []
        for token, prob in token_prob_list:
            if token == '0':
                confidence_score.append(1 - prob)
            else:
                confidence_score.append(prob)
    
    for i,key in enumerate(dim_dict.keys()):
        dim_dict[key] = confidence_score[i]
        dim_answer_dict[key] = dim_answer_list[i]
    
    print(dim_dict)
    return response, dim_dict, dim_answer_dict


def can_convert_to_float(row):#to be updated with 6 mft, and otehrs, is it necccessary?
    last_five = row.iloc[-5:]
    converted = pd.to_numeric(last_five, errors='coerce')
    return not converted.isna().any()

def fix_non_float(result_file, *dims_args):
    result_df = pd.read_csv(result_file)
    err_rows = result_df[~result_df.apply(can_convert_to_float, axis=1)]
    print(err_rows)
    while not err_rows.empty:
        for _, row in err_rows.iterrows():
            stat_id, stat = row["statement_id"], row["statement"]
            # dims_args[0] is the prompt, dims_args[1..] are passed to the corresponding get_dims function
            _ , dim_dict, _ = dims_args[2](
                *dims_args[:2],
                stat,
                *dims_args[3:]
            )
            if all(isinstance(value, float) for value in dim_dict.values()):
                dim_dict["statement"] = stat
                dim_dict["statement_id"] = stat_id
                dim_dict = {"statement_id": dim_dict.pop("statement_id"), "statement": dim_dict.pop("statement"), **dim_dict}        
                result_df.loc[result_df['statement_id'] == stat_id, dim_dict.keys()] = dim_dict.values()
                err_rows = err_rows[err_rows["statement_id"] != stat_id]
    result_df.to_csv(result_file, index=False)

# --- Main ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--to_fix", type=bool, default=False,
                        help="if true, fix rows that cannot be converted to floats")
    parser.add_argument("--api", type=str, default="",
                        help="API server name, i.e. openai, deepseek, etc.")
    parser.add_argument("--deployment_name", default="gpt-4o-mini", type=str, help="model name")
    parser.add_argument("--api_key", type=str, default="",
                        help="API key")
 
    parser.add_argument("--max_workers", type=int, default=5, help="number of parallel workers")

    parser.add_argument("--max_tokens", type=int, default=4096, help="max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0, help="temperature")
    parser.add_argument("--top_p", type=float, default=0.0, help="top-p")
    parser.add_argument("--logprobs", type=bool, default=True, help="logprobs")
    parser.add_argument("--top_logprobs", type=int, default=20, help="top_logprobs")
    parser.add_argument("--input_file", type=str, default="data/MAC_demo.csv",
                        help="data to run moral dimension extraction on")
    parser.add_argument("--prompt_file", type=str, default="one_by_one_prompt.txt",
                        help="file to read prompts from")
    parser.add_argument("--out_file", type=str, default="output/scores.csv",
                        help="file to output scores to")
    parser.add_argument("--answer_file", type=str, default="output/answers.csv",
                        help="file to output model answers to")
    parser.add_argument("--mode", type=str, default="reddits")
    parser.add_argument("--theory", type=str, default="five", choices=["mft","mac","value10","value20","common_morality"],
                        help="select which theory to use: MFT(five or six(with liberty)),Human Value(10 values, 20 values), MAC, or Common Morality")
    parser.add_argument("--with_liberty", type=bool, default=True, help="whether to predict liberty dimension (only for MFT)")
    args = parser.parse_args()


    prompt_text = open(args.prompt_file, "r").read()
    tasks = []
    # Branch based on theory
    if args.theory == "mft":
        moral_stat_list, moral_stat_df = data_process.get_stats(args.input_file, args.mode)
        batch_func = get_dims_mft
        if args.with_liberty:
            prompt_text = open('prompts/demo/MFT_with_liberty_prompt.txt', "r").read()
        for stat, statement_id in zip(moral_stat_list, moral_stat_df["statement_id"].tolist()):
            tasks.append((
                batch_func,
                # common args for five dims: api, prompt, stat, deployment_name, temperature, max_tokens, top_p, logprobs, top_logprobs, with_reason, with_liberty, verbose
                args.api, args.api_key, prompt_text, stat, args.deployment_name, args.temperature, args.max_tokens,
                args.top_p, args.logprobs, args.top_logprobs, args.with_liberty
            ))
    elif args.theory == "mac":

        moral_stat_list, moral_stat_df = data_process.get_stats(args.input_file, args.mode)
        for stat, statement_id in zip(moral_stat_list, moral_stat_df["statement_id"].tolist()):
            tasks.append((
                get_dims_mac,
                # args for mac: prompt, stat, deployment_name, temperature, max_tokens, top_p, logprobs, top_logprobs, with_reason, verbose
                args.api, args.api_key,  prompt_text, stat, args.deployment_name, args.temperature, args.max_tokens,
                args.top_p, args.logprobs, args.top_logprobs#, with_reason, args.verbose
            ))
    elif args.theory in ["value10","value20"]:

        moral_stat_list, moral_stat_df = data_process.get_stats(args.input_file, args.mode)
        if args.theory == "value10":
            prompt_text = open('prompts/demo/Values10_prompt.txt', "r").read()
        elif args.theory == "value20":
            prompt_text = open('prompts/demo/Values20_prompt.txt', "r").read()
        for idx, stat in enumerate(moral_stat_list):
            tasks.append((
                get_dims_humval,
                # args: theory, api, prompt, stat, deployment_name, temperature, max_tokens, top_p, logprobs, top_logprobs, with_reason, verbose, plus extra if any
                args.theory, args.api, args.api_key, prompt_text, stat, args.deployment_name, args.temperature, args.max_tokens,
                args.top_p, args.logprobs, args.top_logprobs
            ))
    elif args.theory == "common_morality":

        moral_stat_list, moral_stat_df = data_process.get_stats(args.input_file, args.mode)
        for stat, statement_id in zip(moral_stat_list, moral_stat_df["statement_id"].tolist()):
            tasks.append((
                get_dims_common_morality,
                # args for common_morality: prompt, stat, deployment_name, temperature, max_tokens, top_p, logprobs, top_logprobs, with_reason, verbose
                args.api, args.api_key,prompt_text, stat, args.deployment_name, args.temperature, args.max_tokens,
                args.top_p, args.logprobs, args.top_logprobs, args.mode
            ))
    else:
        print("Unknown theory selected.")
        return

    print(f"Processing {len(tasks)} examples in batches...")
    all_results = []
    batch_size = 1900
    pause_time = 2
    n_batches = -(-len(tasks)//batch_size)
    for batch_index in range(n_batches):
        start_idx = batch_index * batch_size
        end_idx = start_idx + batch_size
        current_batch = tasks[start_idx:end_idx]
        print(f"Processing batch {batch_index+1}/{n_batches} with {len(current_batch)} examples")
        results = [None] * len(current_batch)
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            future_to_index = {}
            for idx, task in enumerate(current_batch):
                func = task[0]
                func_args = task[1:]
                future = executor.submit(func, *func_args)
                future_to_index[future] = idx
            for future in concurrent.futures.as_completed(future_to_index):
                idx = future_to_index[future]
                try:
                    results[idx] = future.result()
                except Exception as exc:
                    print(f"Task {idx} generated an exception: {exc}")
        all_results.extend(results)
        if batch_index < n_batches - 1:
            print(f"Batch {batch_index+1} complete. Pausing for {pause_time} seconds before next batch.")
            time.sleep(pause_time)
    # Process results and write to CSVs
    save_data = []
    answer_data = []
    for i, (response, dim_dict, answer_dict) in enumerate(all_results):
        dim_dict["statement"] = moral_stat_list[i]
        dim_dict["statement_id"] = i
        dim_dict = {"statement_id": dim_dict.pop("statement_id"), "statement": dim_dict.pop("statement"), **dim_dict}
        answer_dict = {"statement_id": i, "statement": moral_stat_list[i], **answer_dict}
        save_data.append(dim_dict)
        answer_data.append(answer_dict)
    save_df = pd.DataFrame(save_data)
    save_answer_df = pd.DataFrame(answer_data)
    save_df.to_csv(args.out_file, index=False)
    save_answer_df.to_csv(args.answer_file, index=False)
    
if __name__ == "__main__":
    main()