from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the local LLM
model_name = "EleutherAI/gpt-neo-1.3B"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)


documents = SimpleDirectoryReader("~/Data").load_data()
index = VectorStoreIndex.from_documents(documents)

# Configure the query engine to use the local LLM
query_engine = index.as_query_engine(model=model, tokenizer=tokenizer)
# query_engine = index.as_query_engine()
response = query_engine.query("What is the name hometown of Steven Wu?")
print(response)
