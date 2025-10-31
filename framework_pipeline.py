from langchain_community.llms import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from transformers import pipeline



# load small open-source summarization model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=-1)
llm = HuggingFacePipeline(pipeline=summarizer)

prompt = PromptTemplate.from_template("Summarize the following text:\n{text}")
chain = LLMChain(prompt=prompt, llm=llm)

with open("data/article.txt") as f:
    text = f.read()

summary = chain.run({"text": text})
print("🔹 Framework (LangChain) Summary:\n", summary)
