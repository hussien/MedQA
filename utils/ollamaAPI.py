import requests
import json


def dopost(url, json_body):
    try:
        response = requests.post(url, json=json_body)
        if response.status_code == 200:
            return response.content
        else:
            return None
    except Exception as e:
        print('dopost request Error', e)

default_model_options= {
        'temperature': 1,  # Control creativity/randomness
        'top_p': 0.9,
        'top_k': 40,
        'num_predict': 2000,
        'repeat_penalty': 1.1
    }
models_dict={"Llama-3.2-3B-Instruct_FT_MedQA_100S_q8_0":default_model_options,
             "Llama-3.2-3B-Instruct_FT_Lora_MedQA_500S_q8_0":default_model_options,
             "Qwen3-14B_FT_Lora_MedQA_500S_tq2_0":default_model_options,
             "Qwen3-0.6B_FT_Lora_MedQA_F16":default_model_options,
             "Qwen3-14B_FT_Lora_MedQA_500S_q8_0":default_model_options,
             "Qwen3-14B-Q4_K_M":default_model_options,
             "Qwen3-0.6B_FT_Lora_MedQA_q4_k_m":default_model_options,
             "Qwen3-0.6B-Q8_0":default_model_options,
             "Qwen3-4B_FT_Lora_MedQA.Q8_0":default_model_options,
             "Qwen3-8B_FT_Lora_MedQA.Q4_K_M":default_model_options,
             "Qwen3-14B_FT_Lora_MedQA_500S_Q4_K_M":default_model_options,
             "Qwen3-8B_FT_Lora_MedQA_Q8_0":default_model_options,

             "qwen3:0.6b":default_model_options,
             "qwen3:4b":default_model_options,
             "qwen3:8b":default_model_options,
             "medllama3-v20.Q4_K_M":default_model_options,
             "medgemma_4b_1_q8":default_model_options,
             "z":default_model_options,
             }

def query_ollama_dopost(model, prompt, system_prompt=None, temperature=1,inference_api=None):
    # url = "http://192.168.41.218:11434/api/generate"
    # url = "http://206.12.96.43:11434/api/generate"
    dict_ollama_api = {"gpu8": "http://206.12.96.43:11434/api/generate",
                       # "gpu16": "http://206.12.92.147:11434/api/generate",
                       "gpu16": "http://206.12.92.147:22101/api/generate"}
    if inference_api:
        url = inference_api
    else:
        url = dict_ollama_api["gpu8"]

    headers = {"Content-Type": "application/json"}
    if system_prompt:
        prompt = f"<|system|>\n{system_prompt}\n<|user|>\n{prompt}"
    # print("ollama prompt", prompt)
    data = {
        "model": model,
        "prompt": prompt,
        "temperature": temperature,
        "stream": False
    }
    response = requests.post(url, headers=headers, data=json.dumps(data))
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": f"Request failed with status code {response.status_code}"}


def query_ollama_client(model, prompt, system_prompt=None,assistant_prompt=None,temperature=None,inference_api=None):
    from ollama import Client
    # url = "http://192.168.41.218:11434"
    # url = "http://206.12.96.43:11434"
    dict_ollama_api = {"gpu8": "http://206.12.96.43:11434",
                       "gpu16": "http://206.12.92.147:11434"}

    if inference_api:
        url = inference_api
    else:
        url = dict_ollama_api["gpu8"]

    headers = {"Content-Type": "application/json"}
    client = Client(
        host=url,
        headers=headers
    )
    messages = [
        {
            'role': 'system',
            'content': system_prompt if system_prompt else '',
        },
        {
            'role': 'user',
            'content': prompt,
        },
        {
            'role': 'assistant',
            'content': assistant_prompt if assistant_prompt else '',
        },
    ]
    # print("\nmessages=", messages)
    options=models_dict[model]
    if temperature:
        options["temperature"]=temperature
    try:
        response = client.chat(model=model, messages=messages, options=options)
        return json.loads(response.model_dump_json())
    except client.ResponseError as e:
        return {"error": f"{client.ResponseError}"}

def open_ai_request(model, prompt, system_prompt=None,assistant_prompt=None,temperature=None,inference_api=None):
    import openai
    dict_ollama_api = {"gpu8": "http://206.12.96.43:11434",
                       "gpu16": "http://206.12.92.147:22101"}

    if inference_api:
        url=inference_api
    else:
        url = dict_ollama_api["gpu16"]

    client = openai.OpenAI(
        base_url=url,  # "http://<Your api-server IP>:port"
        api_key="sk-no-key-required"
    )

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                'role': 'system',
                'content': system_prompt if system_prompt else '',
            },
            {
                'role': 'user',
                'content': prompt,
            },
            {
                'role': 'assistant',
                'content': assistant_prompt if assistant_prompt else '',
            },
        ]
    )
    try:
        return json.loads(completion.model_dump_json())
    except client.ResponseError as e:
        return {"error": f"{client.ResponseError}"}

def chat(model="o1-mini", prompt_in="", key="", system_prompt=None,assistant_prompt=None,temperature=None,use_ollama=True,inference_api=None):
    if model in ["o1-mini", "gpt-4o-mini"]:
        from openai import OpenAI
        client = OpenAI(api_key=key)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt_in
                        },
                    ],
                }
            ]
        )
        return response.choices[0].message.content, response.usage, response
    elif model == "deepseek-chat":
        from openai import OpenAI
        llm = OpenAI(api_key=key, base_url="https://api.deepseek.com")
        response = llm.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a knoweldge reasoner system"},
                {"role": "user", "content": prompt_in},
            ],
            stream=False
        )
        return response.choices[0].message.content, response.usage, response
    elif model == "deepseek-reasoner":
        from openai import OpenAI
        llm = OpenAI(api_key=key, base_url="https://api.deepseek.com")
        response = llm.chat.completions.create(
            model="deepseek-reasoner",
            messages=[
                {"role": "system", "content": "You are a knoweldge reasoner system"},
                {"role": "user", "content": prompt_in},
            ],
            stream=False
        )
        return response.choices[0].message.content, response.usage, response
    elif model == "gemini-1.5-flash":
        import google.generativeai as genai
        genai.configure(api_key=key)
        model = genai.GenerativeModel(model)
        response = model.generate_content(prompt_in)
        return response.text, response.usage_metadata, response
    # elif model in models_dict :
    else:
        try:
            # response = query_ollama_dopost(model, prompt_in,system_prompt)
            ################# Ollama ##############
            if use_ollama:
                response = query_ollama_client(model, prompt_in, system_prompt,assistant_prompt,temperature,inference_api)
                usage_keys = ['total_duration', 'load_duration', 'prompt_eval_count', 'prompt_eval_duration', 'eval_count','eval_duration']
                return response['message']['content'].split("</think>")[-1].replace("\n", ""), {key: response[key] for key in usage_keys}, response
            else:
                ################ LLamaCPP ################
                response = open_ai_request(model, prompt_in, system_prompt, assistant_prompt, temperature,inference_api)
                usage_keys = ['completion_tokens','prompt_tokens','total_tokens']
                return response["choices"][0]["message"]["content"].split("</think>")[-1].replace("\n", ""), {key: response["usage"][key] for key in usage_keys}, response
        except:
            return None, None, None
