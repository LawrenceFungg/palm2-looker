# Class to create Chroma index from LookML definition
from langchain.docstore.document import Document
from langchain.llms import VertexAI
from langchain.embeddings import TensorflowHubEmbeddings


from typing import List

from langchain.vectorstores import Chroma
from lkml_text_splitter import LkmlTextSplitter


class Indexer:
    def __init__(self, docs: List[Document], llm: VertexAI):
        self.llm = llm
        self.docs = docs

    def run(self):
        # Register the text into the Chroma index
        text_splitter = LkmlTextSplitter(chunk_size=3000)
        texts = text_splitter.split_documents(self.docs)

        url = "https://tfhub.dev/google/universal-sentence-encoder-multilingual/3"
        embeddings = TensorflowHubEmbeddings(model_url=url)
        docsearch = Chroma.from_documents(texts, embeddings)

        return docsearch
