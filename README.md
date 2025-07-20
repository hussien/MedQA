# Medical Question Answring
A Fine tunning + RAG pipline for Medical QA using the MedQA dataset
<div align="center">
<img src="figs/system.png" width="80%"/>
The Medical QA pipelines. A: The SLM Fine-Tuning
pipeline. The SLM is fine-tuned first, then the doctor prompt
the fine-tuned LLM with a medical question. B: The document-
based RAG pipeline. The doctor prompts the non-fine-tuned
LLM, meanwhile, the relevant information for the given ques-
tion is retrieved from a knowledge base and augmented to the
doctor prompt
</div>