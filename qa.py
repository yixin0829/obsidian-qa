"""Ask a question to the notion database."""
import faiss
from langchain import OpenAI
from langchain.chains import VectorDBQAWithSourcesChain
import pickle
import argparse
import time

parser = argparse.ArgumentParser(description='Ask a question to the notion DB.')
parser.add_argument('question', type=str, help='The question to ask the notion DB')
args = parser.parse_args()

start_time = time.perf_counter()

# Load the LangChain. (Change the path to read your own index)
# index = faiss.read_index("docs.index")
index = faiss.read_index("Personal/docs.index")

with open("Personal/faiss_store.pkl", "rb") as f:
    store = pickle.load(f)

store.index = index
chain = VectorDBQAWithSourcesChain.from_llm(llm=OpenAI(temperature=0), vectorstore=store)
load_time = time.perf_counter()

result = chain({"question": args.question})
process_time = time.perf_counter()

print(f"Answer: {result['answer']}")
print(f"Sources: {result['sources']}")

end_time = time.perf_counter()

print(f"Time spent loading: {load_time - start_time:.4f} seconds")
print(f"Time spent processing: {process_time - load_time:.4f} seconds")
print(f"Time spent printing: {end_time - process_time:.4f} seconds")
