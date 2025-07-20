from Utils.ollamaAPI import chat
import ast
import pandas as pd
def llm_as_judge(TP_items_lst,model_name):
  # print(len(TP_items_lst))
  # print(TP_items_lst)
  TP_items_lst= [[a.strip().replace(",","_").replace("(","").replace(")","").replace("{","").replace("}","").replace("[","").replace("]",""),
                  b.strip().replace(",","_").replace("(","").replace(")","").replace("{","").replace("}","").replace("[","").replace("]","")]
                for (a,b) in TP_items_lst]
  pairs_str=str(TP_items_lst).replace("'","")
  print("llm_as_judge pairs_str=",pairs_str)
  lst_str=str(pairs_str).replace('],' , ']\n')
  question_messsage=f"""You act as judge for a given list of True and predicted pair of values . please rank the predicted value compared with true value using two metrics.
1- Exact Match Rule: you compare the two strings after normalizing them and remove any special characters. report 1 if both values are literally and semantically equal and 0 otherwise.
2- Herarical/categorical Match Rule: report 1 if the predicted value is under a subcategory or hierarchically belongs to the true value or is a synonym and 0 otherwise.
Example:
List of pairs: [[music, art], [painter, artist] ,[ football player, soccer player], [ lawyer, judge], [lawyer, player]]
Answer: [[0,1],[0,1],[1,1],[0,1],[0,0]]
My Question List of pairs:
{lst_str}
Note: refine your answers one by one and return Answer for exactly {len(TP_items_lst)} pairs without any explaination and follow strickly the format
Finally: make sure you return only {len(TP_items_lst)} pair of answers without any explanation or thinking details.
The Answer:
"""
  print(question_messsage)
  # response =chat_engine.chat(question_messsage)
  print("model_name=",model_name)
  response,usage,full_response=chat(model=model_name,prompt_in=question_messsage)
  print("llm_as_judge response","[["+response.split("[[")[-1].split("]]")[0]+"]]")
  lst=None
  try:
    lst=ast.literal_eval("[["+response.split("[[")[-1].split("]]")[0]+"]]")
  except:
    print("The eval list is not well formated")
  print("Eval:",lst)
  return lst,response,usage,full_response

def eval_predictions_Exact(ground_truth_dict,predictd_WOC_dict):
    WOC_acc_res={}
    merged_df_res={}
    for idx, (k,v) in enumerate(ground_truth_dict.items()):
      col_title=k.split('-')[1]
      predictd_WOC_dict[k][0].columns=["target",col_title]
      predictd_WOC_dict[k][0]['target']=predictd_WOC_dict[k][0]['target'].apply(lambda x:str(x).strip())
      # print(predictd_WC_dict[k][0].columns)
      ground_truth_dict[k]['target']=ground_truth_dict[k]['target'].apply(lambda x:str(x).strip())
      # print(ground_truth_dict[k].columns)
      merged_df=pd.merge(ground_truth_dict[k], predictd_WOC_dict[k][0], left_on='target',right_on='target', how='inner')
      if len(merged_df)==0:
          merged_df=pd.merge(ground_truth_dict[k], predictd_WOC_dict[k][0], left_on='target_txt',right_on='target', how='inner')
      # print(merged_df.columns)
      if len(merged_df)>0:
        merged_df[col_title+"_txt"]=merged_df[col_title+"_txt"].apply(lambda x: str(x).replace("_"," "))
        merged_df[col_title+"_y"]=merged_df[col_title+"_y"].apply(lambda x: str(x).replace("_"," "))
        true_count=0
        for idx, row in merged_df.iterrows():
          if (str(row[col_title+"_y"]).strip().lower() in str(row[col_title+"_txt"]).strip().lower() or
             str(row[col_title+"_txt"]).strip().lower() in str(row[col_title+"_y"]).strip().lower()):
            true_count+=1

        WOC_acc_res[k]=[true_count/len(merged_df),0, true_count]
        merged_df_res[k]=merged_df
      else:
        WOC_acc_res[k]=[0,0, 0]
        merged_df_res[k]=None
    return WOC_acc_res,merged_df_res

def eval_LLM_WC(ground_truth_dict,predictd_WC_dict):
  WC_acc_res={}
  merged_df_res={}
  for idx, (k,v) in enumerate(ground_truth_dict.items()):
    col_title=k.split('-')[1]
    predictd_WC_dict[k][0]['target']=predictd_WC_dict[k][0]['target'].apply(lambda x:str(x).strip())
    merged_df=pd.merge(ground_truth_dict[k], predictd_WC_dict[k][0], left_on='target_txt',right_on='target', how='inner')
    # print(merged_df.columns)
    if len(merged_df)>0:
      merged_df[col_title+"_txt"]=merged_df[col_title+"_txt"].apply(lambda x: str(x).replace("_"," "))
      merged_df[col_title+"_y"]=merged_df[col_title+"_y"].apply(lambda x: str(x).replace("_"," "))
      ####################### LLM Judge ##########
      pairs=list(zip(list(merged_df[col_title+"_txt"].values),list(merged_df[col_title+"_y"].values)))
      res,response,usage,full_response=llm_as_judge(pairs)
      l1,l2=zip(*res)
      merged_df=merged_df.head(len(res))
      merged_df["is_true_pred"]=list(l1)
      merged_df["pred_similarity_score"]=list(l2)
      ####################### hardcoded Metric ###################
      # merged_df["is_true_pred"]= merged_df.apply(lambda row: compare_strings(row,col_title+"_txt",col_title+"_y"),axis=1)
      # merged_df["pred_similarity_score"]= merged_df.apply(lambda row: get_similarity_score(row,col_title+"_txt",col_title+"_y"),axis=1)
      WC_acc_res[k]=[sum(merged_df["is_true_pred"])/len(merged_df),sum(merged_df["pred_similarity_score"])/len(merged_df), sum(merged_df["is_true_pred"])]
      merged_df_res[k]=merged_df
    else:
      WC_acc_res[k]=[0,0, 0]
      merged_df_res[k]=None
    # print(f"""{k} task: String Matching Accuracy={sum(merged_df["is_true_pred"])/len(merged_df)},Pred Similarity Score ={sum(merged_df["pred_similarity_score"])/len(merged_df)}, True answers Count={sum(merged_df["is_true_pred"])}""")
  return WC_acc_res,merged_df_res


if __name__ == '__main__':
    lst=[['Austria',' Austria'],['Arjuna Award',' Gold Medal'],['Arjuna Award',' Gold Medal'],['Order of Friendship',' Gold Medal'],['Order of Friendship',' Gold Medal'],['Order of Friendship',' Gold Medal'],['Order of the British Empire',' Gold Medal'],['100 Women BBC',' Gold Medal'],['100 Women BBC',' Gold Medal'],['100 Women BBC',' Award for Best Actress'],['100 Women BBC',' Award for Best Designer'],['100 Women BBC',' Award for Best Athlete'],['Guggenheim Fellowship',' Award for Best Director'],['Order of Friendship',' Gold Medal'],['Order of Friendship',' Gold Medal'],['Order of Friendship',' Gold Medal'],['Order of Friendship',' Gold Medal'],['Order of Friendship',' Gold Medal'],['Order of Honour Russia',' Gold Medal'],['Order of the British Empire',' Award for Best Newcomer']]
    res_lst,response,usage,full_response=llm_as_judge(lst)
    print(res_lst)