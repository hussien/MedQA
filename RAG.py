import os
from utils.ollamaAPI import chat
from datetime import datetime
import numpy as np
import pandas as pd
from datasets import Dataset
from tqdm.notebook import tqdm
from langchain.docstore.document import Document as LangchainDocument
from tqdm import tqdm
import pandas as pd
from transformers import AutoTokenizer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pickle
import faiss
from sentence_transformers import SentenceTransformer
from utils.utils import get_test_question,get_prompts
def load_data(directory):
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):  # Assuming the files are .txt files
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
                doc_title = filename[:-4]  # Assuming the title is the filename without .txt
                doc_text = file.read()
                documents.append([doc_title, doc_text])
    return pd.DataFrame(documents, columns=["title", "text"])
MARKDOWN_SEPARATORS = [
        "\n#{1,6} ",
        "```\n",
        "\n\\*\\*\\*+\n",
        "\n---+\n",
        "\n___+\n",
        "\n\n",
        "\n",
        " ",
        "",
    ]
def split_documents(chunk_size,knowledge_base,tokenizer_name):
        """
        Split documents into chunks of maximum size `chunk_size` tokens and return a list of documents.
        """
        text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            AutoTokenizer.from_pretrained(tokenizer_name),
            chunk_size=chunk_size,
            chunk_overlap=int(chunk_size / 10),
            add_start_index=True,
            strip_whitespace=True,
            separators=MARKDOWN_SEPARATORS,
        )

        docs_processed = []
        for doc in knowledge_base:
            docs_processed += text_splitter.split_documents([doc])

        # Remove duplicates
        unique_texts = {}
        docs_processed_unique = []
        for doc in docs_processed:
            if doc.page_content not in unique_texts:
                unique_texts[doc.page_content] = True
                docs_processed_unique.append(doc)

        return docs_processed_unique
def process_docs(docs_directory, proccessed_pickle_path=""):
    documents_df = load_data(docs_directory)
    ds = Dataset.from_pandas(documents_df)
    RAW_KNOWLEDGE_BASE = [
        LangchainDocument(page_content=doc["text"], metadata={"source": doc["title"]})
        for doc in tqdm(ds)]

    # We use a hierarchical list of separators specifically tailored for splitting Markdown documents
    # This list is taken from LangChain's MarkdownTextSplitter class
    docs_processed = []
    if os.path.exists(proccessed_pickle_path):
        with open(proccessed_pickle_path, 'rb') as f:
            docs_processed = pickle.load(f)
    else:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,  # The maximum number of characters in a chunk: we selected this value arbitrarily
            chunk_overlap=100,  # The number of characters to overlap between chunks
            add_start_index=True,  # If `True`, includes chunk's start index in metadata
            strip_whitespace=True,  # If `True`, strips whitespace from the start and end of every document
            separators=MARKDOWN_SEPARATORS,
        )
        for doc in RAW_KNOWLEDGE_BASE:
            docs_processed += text_splitter.split_documents([doc])
        with open(proccessed_pickle_path, 'wb') as f:
            pickle.dump(docs_processed, f)
    return docs_processed

data_path = 'data/'
docs_directory = data_path+'data_clean/textbooks/en'
if __name__ == '__main__':
    docs_processed=process_docs(docs_directory,data_path+"MedQA_en_documents.pkl")
    # Load the model
    model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
    ############## Generate Docs Embeddings #####################
    embds = []
    if os.path.exists(data_path + 'docs_emb_qwen3.pkl'):
        with open(data_path + 'docs_emb_qwen3.pkl', 'rb') as f:
            embds = pickle.load(f)
    else:
        for doc in tqdm(docs_processed):
            embds.append(model.encode(doc.metadata['source'] + "\n" + doc.page_content))
        with open(data_path + 'docs_emb_qwen3.pkl', 'wb') as f:
            pickle.dump(embds, f)
    ################# build Faiss VDB ##############
    KNOWLEDGE_VECTOR_DATABASE = faiss.IndexFlatIP(embds[0].shape[0])  # build the index
    print(KNOWLEDGE_VECTOR_DATABASE.is_trained)
    KNOWLEDGE_VECTOR_DATABASE.add(np.array(embds))  # add vectors to the index
    print(KNOWLEDGE_VECTOR_DATABASE.ntotal)
    ################ Process RAG ####################
    user_prompt = None
    system_prompt = "Use the supportive information given to refine your answer."
    # assistant_prompt=" Avoid not return explaination, thinking or analysis.\nReturn the correct answer only."
    assistant_prompt = None
    test_size = 100
    llm_test_dataset = get_test_question(test_size=test_size)
    print(llm_test_dataset)
    results = []
    model_name = "qwen3:8b"
    model_name = "qwen3:4b"
    model_name = "qwen3:0.6b"
    model_name = "Qwen3-14B-Q8_0"
    model_name= "Qwen3-14B_FT_Lora_MedQA_500S_Q4_K_M"
    # model_name="Qwen3-0.6B_FT_Lora_MedQA_q4_k_m"
    # model_name="Qwen3-4B_FT_Lora_MedQA.Q8_0"
    # model_name="Qwen3-8B_FT_Lora_MedQA.Q4_K_M"
    if "_FT_" in model_name or True:
        use_ollama = False
    else:
        use_ollama = True
    print(f"model_name={model_name}\tuse_ollama={use_ollama}")
    for idx, row in enumerate(llm_test_dataset):
        ########### get RAG top-1 Doc (Search FAISS ############
        query_vector = model.encode([row["question"]])
        k = 3  # Number of nearest neighbors to retrieve
        distances, indices = KNOWLEDGE_VECTOR_DATABASE.search(query_vector, k)
        print(f"top-k docs={indices} with distances={distances}")
        row["rag_doc"]=docs_processed[int(distances[0][0])].page_content[0:1500]
        user_prompt, system_prompt, answer, answer_idx = get_prompts(row)
        response, usage, full_response = chat(model=model_name,
                                              prompt_in=user_prompt, key="",
                                              system_prompt=system_prompt,
                                              assistant_prompt=assistant_prompt,
                                              temperature=None, use_ollama=use_ollama)
        model_answer = ""
        try:
            model_answer = response.split("Answer")[-1].replace("'", "").split(" ")[0].replace(":", "")[0]
        except:
            model_answer = response.split("Answer")[-1].replace("'", "").split(" ")[0]

        results.append([idx, model_answer, answer_idx, str(usage)])
        print(f"Q_idx:{idx} \t Pred Answer={response.split("Answer")[-1]}\t\t Real Answer idx={answer_idx}")
    results_df = pd.DataFrame(results, columns=["Q_idx", "pred", "real", "usage"])
    results_df["Correct_Ans"] = results_df.apply(lambda x: 1 if x["real"] == x["pred"] else 0, axis=1)
    results_df.to_csv("RAG_"+model_name + f"_results_{test_size}_ts_{datetime.now()}.csv", index=False)
    print(f"Accuracy={results_df["Correct_Ans"].sum() / len(results_df["Correct_Ans"])}")

