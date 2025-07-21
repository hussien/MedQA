from unsloth import FastLanguageModel
from trl import SFTConfig, SFTTrainer
from datasets import load_dataset
import torch
import argparse
def format_chat_template(row):
    system = "You are a professional, highly experienced doctor professor. \n please answers the patients' questions using only one of the options in the brackets."
    patient_case = "#Patient Case:\n" + row["question"]
    Choices = "#Choices:\n" + "\n".join([str(k) + ": " + str(v) for k, v in row["options"].items()])
    Answer = "#Answer\n:" + row["answer_idx"] + ": " + row["answer"] + "\n"
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": patient_case + "\n" + Choices},
        {"role": "assistant", "content": Answer}
    ]
    row['text'] = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    return row
def save_model(model,model_name):
    model_name = model_name.split("/")[-1] + "_FT_Lora_MedQA"
    model.save_pretrained(model_name)  # Local saving
    tokenizer.save_pretrained(model_name)
    
    # Save to 8bit Q8_0
    if False: model.save_pretrained_gguf(model_name + "_q8", tokenizer, quantization_method="q8_0")
    # Remember to go to https://huggingface.co/settings/tokens for a token!
    # And change hf to your username!
    if False: model.push_to_hub_gguf("hf/model", tokenizer, token="", quantization_method="q8_0")

    # Save to 16bit GGUF
    if False: model.save_pretrained_gguf(model_name + "_fb16", tokenizer, quantization_method="f16")
    if False: model.push_to_hub_gguf("hf/model", tokenizer, quantization_method="f16", token="")

    # Save to q4_k_m GGUF
    if True: model.save_pretrained_gguf(model_name + "_q4_k_m", tokenizer, quantization_method="q4_k_m")
    if False: model.push_to_hub_gguf("hf/model", tokenizer, quantization_method="q4_k_m", token="")

    # Save to multiple GGUF options - much faster if you want multiple!
    if False:
        model.push_to_hub_gguf(
            "hf/model",  # Change hf to your username!
            tokenizer,
            quantization_method=["q4_k_m", "q8_0", "q5_k_m", ],
            token="",  # Get a token at https://huggingface.co/settings/tokens
        )
def unsloth_finetune(model_name="unsloth/Qwen3-8B",dataset_name="GBaker/MedQA-USMLE-4-options",max_steps=500):
    max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
    dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+    
    max_seq_length = 2048
    dtype = None # Auto detection of dtype
    load_in_4bit = True
    model, tokenizer = FastLanguageModel.from_pretrained(
     model_name=model_name,
     max_seq_length = max_seq_length,
     dtype = dtype,
     load_in_4bit = load_in_4bit,)    
    
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = 3407,
        use_rslora = False,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
    )
    
    
    # dataset = load_dataset("medalpaca/medical_meadow_medqa", split="train")
    llm_train_dataset = load_dataset(dataset_name, split="train")   
    llm_train_dataset = llm_train_dataset.map(format_chat_template, num_proc=8, )   
    
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = llm_train_dataset,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        dataset_num_proc = 2,
        packing = False, # Can make training 5x faster for short sequences.
        args = SFTConfig(
            per_device_train_batch_size = 4,
            gradient_accumulation_steps = 16,
            warmup_steps = 0,
            max_steps = max_steps,
            num_train_epochs = 3, # For longer training runs!
            learning_rate = 2e-5,
            logging_steps = 10,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "cosine",
            seed = 3407,
            output_dir = "outputs",
            report_to = "none", # Use this for WandB etc
        ),
    )
    
    # @title Show current memory stats
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")    
    trainer_stats = trainer.train()    
    # @title Show final memory and time stats
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory / max_memory * 100, 3)
    lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
    print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
    print(
        f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training."
    )
    print(f"Peak reserved memory = {used_memory} GB.")
    print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
    print(f"Peak reserved memory % of max memory = {used_percentage} %.")
    print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")
    return model
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GLOW-QA')
    parser.add_argument('--model_name', type=str, default="unsloth/Qwen3-8B",help="The LLM version of the model")
    parser.add_argument('--dataset_name', type=str, default='GBaker/MedQA-USMLE-4-options',help="The training set")
    parser.add_argument('--max_steps', type=int, default=500, help='number of finetunning steps')
    args = parser.parse_args()
    model=unsloth_finetune(args.model_name,args.dataset_name,args.max_steps)
    save_model(model,args.model_name)
    # !zip -r {model_name}.zip {model_name}
    # !zip -0 -r /content/Qwen3-4B_FT_Lora_MedQA_q4_k_m.zip /content/Qwen3-4B_FT_Lora_MedQA_q4_k_m
    