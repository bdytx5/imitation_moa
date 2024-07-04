import json
import os
import subprocess
import torch
from tqdm import tqdm
from llama_index import VectorStoreIndex, ServiceContext, StorageContext, load_index_from_storage
from llama_index.schema import Document
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.retrievers import VectorIndexRetriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.postprocessor import SimilarityPostprocessor
from llm2vec import LLM2Vec

# Load the chunked WikiSource data
with open("chunked_wikisource_data.json", 'r', encoding='utf-8') as file:
    chunked_data = json.load(file)

# Initialize the llm2vec model
l2v = LLM2Vec.from_pretrained(
    "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
    peft_model_name_or_path="McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-unsup-simcse",
    device_map="cuda" if torch.cuda.is_available() else "cpu",
    torch_dtype=torch.bfloat16,
)

# Function to create document from JSON item
def create_document_from_json_item(json_item):
    document = Document(text=json_item['text'], metadata=json_item)
    return document

# Function to generate embeddings for document using llm2vec
# def generate_embeddings_for_document(document):
#     embedding = l2v.encode([document.text])[0]
#     return embedding


def generate_embeddings_for_document(document, model_name="BAAI/bge-small-en-v1.5"):
    embed_model = HuggingFaceEmbedding(model_name=model_name)
    embeddings = embed_model.get_text_embedding(document.text)
    return embeddings



# Generate documents and embeddings
documents = []
for item in tqdm(chunked_data, desc="Generating documents and embeddings"):
    document = create_document_from_json_item(item)
    document.embedding = generate_embeddings_for_document(document)
    documents.append(document)

# Create and persist the vector index
if not os.path.exists("./index"):
    service_context = ServiceContext.from_defaults(llm=None, embed_model='local')
    index = VectorStoreIndex.from_documents(documents, service_context=service_context)
    index.storage_context.persist(persist_dir="./index")
else:
    storage_context = StorageContext.from_defaults(persist_dir="./index")
    service_context = ServiceContext.from_defaults(llm=None, embed_model='local')
    index = load_index_from_storage(storage_context, service_context=service_context)

# # Function to query the index
# def query_index(query):
#     retriever = VectorIndexRetriever(index=index, similarity_top_k=10)
#     response_synthesizer = get_response_synthesizer(service_context=service_context)
#     query_engine = RetrieverQueryEngine(retriever=retriever, response_synthesizer=response_synthesizer, node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.6)])
#     response = query_engine.query(query)
#     return response

# # Example query
# query = "chicken appetizers"
# response = query_index(query)
# print(response)

# # Save the QA dataset to a JSON file
# output_file_path = "./qa_dataset.json"
# with open(output_file_path, 'w', encoding='utf-8') as file:
#     json.dump(response, file, ensure_ascii=False, indent=4)

# print(f"QA dataset has been saved to {output_file_path}")
