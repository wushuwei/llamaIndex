from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama

# Load the SentenceTransformer model
model_name = "BAAI/bge-small-en-v1.5"

# Set the embedding model to use Hugging Face embeddings
embedding_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")


# Apply the Hugging Face embedding model to the Settings class
Settings.embed_model = embedding_model

# Settings.llm = Ollama(model="llama2", request_timeout=60.0)
Settings.llm = Ollama(model="deepseek-r1", request_timeout=600.0)

# load data, prepare index
documents = SimpleDirectoryReader("~/Data").load_data()
index = VectorStoreIndex.from_documents(documents)

query_engine = index.as_query_engine()

response = query_engine.query("what is the hometown for steven wu?")

print(response)
