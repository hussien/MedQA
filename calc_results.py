import pandas as pd
import os
import ast
def get_avg(df,column_name):
    df[column_name] = df["usage_dict"].apply(lambda x: x[column_name] if column_name in x else 0)
    avg = df[column_name].sum() / len(df)
    return avg
if __name__ == "__main__":
    results=[]
    for filename in os.listdir("results/"):
        if filename.endswith(".csv"):  # Assumin
            print(f"filename={filename}")# g the files are .txt files
            df=pd.read_csv(os.path.join("results/", filename))
            try:
                if "Correct_Ans" in df.columns:
                    acc=df["Correct_Ans"].sum()/len(df)
                if "usage" in df.columns:
                    df["usage_dict"] = df["usage"].apply(lambda x: ast.literal_eval(str(x)))
                    avg_completion_tokens=get_avg(df,"completion_tokens")
                    avg_prompt_tokens = get_avg(df, "prompt_tokens")
                    avg_total_tokens = get_avg(df, "total_tokens")
                    avg_eval_count=get_avg(df, "eval_count")
                    avg_total_duration=get_avg(df, "total_duration")
                    results.append([filename,acc,avg_completion_tokens,avg_prompt_tokens,avg_total_tokens,avg_eval_count,avg_total_duration])
            except:
                continue
    res_df=pd.DataFrame(results,columns=["filename","acc","avg_completion_tokens","avg_prompt_tokens",
                                         "avg_total_tokens","avg_eval_count","avg_total_duration"])
    res_df.to_csv("results.csv",index=False)