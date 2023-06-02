# # Overview
# 
# Question answering over documents consists of four steps:
# 1. Create an index
# 2. Create a Retriever from that index
# 3. Create a question answering chain
# 4. Ask questions!

# # 1 Create an index
# Okay, so what’s actually going on? How is this index getting created? A lot of the magic is being hid in this VectorstoreIndexCreator. What is this doing?
# 
# There are three main steps going on after the documents are loaded (inside `VectorstoreIndexCreator`):
# 1. Splitting documents into chunks
# 2. Creating embeddings for each document
# 3. Storing documents and embeddings in a vectorstore

from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.document_loaders import DirectoryLoader
from langchain.indexes import VectorstoreIndexCreator
from pathlib import Path

loader = DirectoryLoader("Obsidian_DB/", glob="**/*.md", show_progress=True)
docs = loader.load()
print(f"len of docs {len(docs)}")

# Each Document object has two fields: page_content and metadata
dict(docs[0])

# By default, LangChain uses Chroma as the vectorstore to index and search embeddings.
index = VectorstoreIndexCreator().from_loaders([loader]) # return `VectorStoreIndexWrapper`

# Check package used for creating the vector store
print(index.vectorstore)
# Check Retriever (how to find answer)
print(index.vectorstore.as_retriever())

# # 2 Create a Retriever from index
# 
# Logic is included in `query_with_sources`

# # 3 Create a question answering chain
# 
# Used `RetrievalQAWithSourcesChain`

# # 4 Ask questions!

# By default query_with_sources uses OpenAI text-davinci-001 to generate answer
index.query_with_sources("What's a vector storage?", llm=OpenAI(temperature=0))

index.query_with_sources("What's the difference between word error rate (WER) and BLEU score?")
