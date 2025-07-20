from utils.ollamaAPI import chat
from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from datetime import datetime
from utils.utils import get_test_question,get_prompts

if __name__ == '__main__':
    user_prompt=None
    system_prompt=None
    # assistant_prompt=" Avoid not return explaination, thinking or analysis.\nReturn the correct answer only."
    assistant_prompt = None
    test_size=100
    llm_test_dataset=get_test_question(test_size=test_size)
    print(llm_test_dataset)
    results=[]
    # model_name="Llama-3.2-3B-Instruct_FT_MedQA_100S_q8_0"
    # model_name="Llama-3.2-3B-Instruct_FT_Lora_MedQA_500S_q8_0"
    # model_name="Qwen3-0.6B_FT_Lora_MedQA_F16"
    # model_name = "Qwen3-14B_FT_Lora_MedQA_500S_q8_0"
    # model_name ="Qwen3-14B-Q8_0"
    # model_name="Qwen3-0.6B-Q8_0:latest"
    # model_name = "z"
    model_name="qwen3:0.6b"
    model_name="qwen3:4b"
    model_name = "qwen3:8b"

    # model_name="Qwen3-0.6B_FT_Lora_MedQA_F16"
    # model_name="Qwen3-0.6B_FT_Lora_MedQA_q4_k_m"
    # model_name="Qwen3-4B_FT_Lora_MedQA.Q8_0"
    # model_name="Qwen3-8B_FT_Lora_MedQA.Q4_K_M"
    model_name="Qwen3-14B_FT_Lora_MedQA_500S_Q4_K_M"
    if "_FT_" in model_name or True:
        use_ollama=False
    else:
        use_ollama = True
    print(f"model_name={model_name}\tuse_ollama={use_ollama}")
    for idx,row in enumerate(llm_test_dataset):
        user_prompt,system_prompt,answer,answer_idx=get_prompts(row)
        response, usage, full_response = chat(model=model_name,
                                              prompt_in=user_prompt, key="",
                                              system_prompt=system_prompt,
                                              assistant_prompt=assistant_prompt,
                                              temperature=None,use_ollama=use_ollama)
        model_answer=""
        try:
            model_answer=response.split("Answer")[-1].replace("'", "").split(" ")[0].replace(":", "")[0]
        except:
            model_answer = response.split("Answer")[-1].replace("'", "").split(" ")[0]

        results.append([idx,model_answer,answer_idx,str(usage)])
        print(f"Q_idx:{idx} \t Pred Answer={response.split("Answer")[-1]}\t\t Real Answer idx={answer_idx}")
    results_df=pd.DataFrame(results,columns=["Q_idx","pred","real","usage"])
    results_df["Correct_Ans"]=results_df.apply(lambda x: 1 if x["real"]==x["pred"] else 0,axis=1)
    results_df.to_csv(model_name+f"_results_{test_size}_ts_{datetime.now()}.csv",index=False)
    print(f"Accuracy={results_df["Correct_Ans"].sum()/len(results_df["Correct_Ans"])}")

