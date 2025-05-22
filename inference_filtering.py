import json
import torch
import transformers
import openai
import fire
import argparse
from functools import partial
from utils.formatting_mcq import formatting_mcq

# python inference_filtering.py --method DeRAGEC --data_type cv
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--method', type=str, default="DeRAGEC",
                    required=False, help='Method to use.')
parser.add_argument('-d', '--data_type', type=str, default="cv",
                    required=False, help='Data to use.')
args = parser.parse_args()

class Config():
    method: str = args.method # RAGEC|DeRAGEC
    data_type: str = args.data_type # cv|stop
    model_id: str = "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4" # hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4|gpt-4o-mini-2024-07-18
    asr_model: str = "whisper-turbo"
    batch_size: int = 1
    temperature: float = 0.2 # LLM temperature
    max_new_tokens: int = 1024

def main(
    cfg = Config(),
    verbose=True,
):

    # 1. load model
    if "gpt" in cfg.model_id:
        # Enter your personal GPT API key here
        api_key = open('../api.txt', 'r').read()
        client = openai.OpenAI(api_key=api_key)  # Create a client instance
        model_type = cfg.model_id
        
    else:
        model_type = cfg.model_id.split("/")[1]
        pipeline = transformers.pipeline(
            "text-generation",
            model=cfg.model_id,
            model_kwargs={"torch_dtype": torch.float16},
            device_map="cuda",
            batch_size=cfg.batch_size,
        )

        tokens = pipeline.model.config.eos_token_id
        pipeline.tokenizer.pad_token_id = tokens[0] if isinstance(tokens, list) else tokens
        pipeline.tokenizer.padding_side = "left"

        pipeline.model.eval()

        terminators = [
            pipeline.tokenizer.eos_token_id,
            pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]

    # 2. load input file
    datapoint = load_data_from_json(f"examples/{cfg.data_type}_examples.jsonl")[0]
    
    fewshot_examples = open(f"fewshots/{cfg.method}_{cfg.asr_model}_filtering.txt", "r")
    fewshot_examples = fewshot_examples.read()
    fewshot_examples = "<input>".join(fewshot_examples.split("<input>")[:5]) # Number of few-shot examples included in the prompt
    
    if cfg.method == "RAGEC":
        template = open(f"templates/template_MCQ.txt", "r")
    if cfg.method == "DeRAGEC":
        template = open(f"templates/template_MCQ+reasoning.txt", "r")
    template = template.read()
    
    # NE dictionary
    f_dict = open(f"examples/dictionary.json", "r")
    f_dict = json.load(f_dict)
    
    template = partial(template.format, fewshot_examples=fewshot_examples) 
    datapoint["filtered-NE"] = {}
    datapoint["r-NE"] = {}

    # 3. inference
    for q in datapoint["p-query"]:
        input = format_prompt(template, datapoint, q, f_dict)
        
        system_prompt = "You are a helpful assistant that selects the correct Named Entity to fill in the blank space."
        prompt = [
                {"role": "system", "content": f"{system_prompt}"},
                {"role": "user", "content": f"{input}"},
            ]
        
        file = open(f"output/{cfg.data_type}_examples_{cfg.method}_filtered_{model_type}.jsonl", "a")
                
        if "gpt" in cfg.model_id:
            response = client.chat.completions.create(
                model=model_type,  
                messages=prompt,
                temperature=cfg.temperature,
                max_tokens=cfg.max_new_tokens,
            )
            assistant_response = response.choices[0].message.content
            
        else:
            response = pipeline(
                input,
                max_new_tokens=cfg.max_new_tokens,
                eos_token_id=terminators,
                do_sample=True, 
                temperature=cfg.temperature,
            )
            assistant_response = response[0]["generated_text"].split("[Test Case]")[1].split("<output>")[1]
        
        if verbose:       
            test_case = input.split("[Test Case]")[1]             
            print(test_case)
            print(assistant_response)
            print("-" * 50)
        else:
            print(assistant_response)
        
        if cfg.method == "RAGEC":
            datapoint["filtered-NE"][q] = assistant_response.split("Answer:")[1].split(":")[1].strip()
            datapoint["r-NE"][q] = None
                
        elif cfg.method == "DeRAGEC":
            datapoint["filtered-NE"][q] = assistant_response.split("<answer>")[1].split("</answer>")[0].split(":")[1].strip()
            datapoint["r-NE"][q] = assistant_response.split("<think>")[1].split("</think>")[0].strip()
            
        json.dump(datapoint, file, ensure_ascii=False)
        file.write("\n")
    

def format_prompt(template, datapoint, q, f_dict):
    
    masked_sentence, options = formatting_mcq(datapoint, q, f_dict)
    prompt = template(masked_sentence=masked_sentence, options=options)
                
    return prompt
                        
def load_data_from_json(jsonl_path):
    with open(jsonl_path, "r") as f:
        jsonl_data = list(f)

    data = []
    for pair in jsonl_data:
        sample = json.loads(pair)
        data.append(sample)

    return data

                        
if __name__ == "__main__":
    main()