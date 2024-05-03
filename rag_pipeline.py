import os
import together
# import shutil
import time
# import textwrap
from typing import Any, Dict

from pydantic import Extra, Field, root_validator, model_validator
from langchain_core.documents import Document
# from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
# from langchain.llms.utils import enforce_stop_tokens
from langchain.utils import get_from_dict_or_env
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
import configs
# from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
# from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader
# from langchain.embeddings import HuggingFaceBgeEmbeddings
from dotenv import load_dotenv

# set your API key
load_dotenv()
os.environ['TOGETHER_API_KEY']
together.api_key = os.environ["TOGETHER_API_KEY"]



################################################################################
# split documents into chunks, create embeddings, store embeddings in chromaDB #
################################################################################
def chunk_and_embed():
    src_dir = f'./studies'
    dst_dir = f'./rag_data/data'
    files = [f for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f))]
    persist_directory = f'db'
    t1 = time.perf_counter()
    try:
        """split documents into chunks, create embeddings, store embeddings in chromaDB"""
        print("4......inside chunk_and_embed try statement")
        # Load each text file as a separate document
        documents = []
        for file in files:
            file_path = os.path.join(src_dir, file)
            with open(file_path, 'r') as f:
                text = f.read()
                document = Document(page_content=text, metadata={'source': file_path})
                documents.append(document)
        
        print(f' =========== number of documents {len(documents)} =========== ')
     
        Chroma.from_documents(documents=documents,
                              embedding=configs.embedding,
                              persist_directory=persist_directory)
        # chunk_size = 2000
        # chunk_overlap=20
        # print(f"split TXT files")
        # print(os.listdir(src_dir))
        # loader = DirectoryLoader(src_dir, glob="./*.txt", loader_cls=TextLoader)

        # documents = loader.load()
        # print(f' =========== number of documents {len(documents)} =========== ')
        # #splitting the text into
        # text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        # texts = text_splitter.split_documents(documents)
        # print(f'=========== number of chunks {len(texts)} ===========')
     
        # Chroma.from_documents(documents=texts,
        #                                 embedding=configs.embedding,
        #                                 persist_directory=persist_directory)

        t2 = time.perf_counter()
        print(f'time taken to embed {len(documents)} chunks:',t2-t1)
        print(f'time taken to embed {len(documents)} chunks:,{(t2-t1)/60} minutes')
        print(f'time taken to embed {len(documents)} chunks:,{((t2-t1)/60)/60} hours')

        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)

        print(f"Moved {len(files)} files from {src_dir} to {dst_dir}.")
        print(f"Files moved: {files}")
        print("\n".join(files))
        
        return f'time taken to embed {len(documents)} chunks:,{(t2-t1)/60} minutes'

    except Exception as e:
        for file in files:
            src_file_path = os.path.join(src_dir, file)
            os.remove(src_file_path)
        print(f"ERROR NOOOOOOOOO: {e}")

