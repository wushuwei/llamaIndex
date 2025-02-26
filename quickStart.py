from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings

# global default
# By default, LlamaIndex will use text-embedding-ada-002
Settings.embed_model = OpenAIEmbedding()

# Settings.embed_model = HuggingFaceEmbedding(
#     model_name="BAAI/bge-small-en-v1.5"
# )


documents = SimpleDirectoryReader("~/Data").load_data()

index = VectorStoreIndex.from_documents(documents)

query_engine = index.as_query_engine()

response = query_engine.query("query string")

print(response)