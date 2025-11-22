from typing import List, Dict, Any
from pathlib import Path
import sys

CURRENT_FILE = Path(__file__).resolve()
SRC_DIR = CURRENT_FILE.parent.parent
PROJECT_ROOT = SRC_DIR.parent

sys.path.insert(0, str(SRC_DIR))

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document

from utils.logger import setup_logger

logger = setup_logger(__name__)

class RAGRetriever:
    """Retrieve relevant documents for RAG"""
    
    def __init__(self, vectorstore_path=None, k=3):
        if vectorstore_path is None:
            vectorstore_path = SRC_DIR / "rag_index" / "vectorstore"
        
        self.vectorstore_path = Path(vectorstore_path)
        self.k = k
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        self.vectorstore = None
        self.load_vectorstore()
    
    def load_vectorstore(self):
        """Load vector store"""
        if not (self.vectorstore_path / "chroma.sqlite3").exists():
            logger.error(f"Vector store not found at {self.vectorstore_path}")
            logger.info("Run: python src/rag_index/build_index.py")
            raise FileNotFoundError("Vector store not built yet")
        
        self.vectorstore = Chroma(
            persist_directory=str(self.vectorstore_path),
            embedding_function=self.embeddings
        )
        logger.info("Vector store loaded")
    
    def retrieve(self, query: str, k=None):
        """Retrieve relevant documents"""
        if k is None:
            k = self.k
        
        docs = self.vectorstore.similarity_search(query, k=k)
        return docs
    
    def get_relevant_context(self, query: str, k=None):
        """Get context with sources"""
        docs = self.retrieve(query, k)
        
        context = "\n\n".join([doc.page_content for doc in docs])
        sources = [doc.metadata.get('source', 'Unknown') for doc in docs]
        
        return {
            'context': context,
            'sources': list(set(sources)),
            'num_docs': len(docs)
        }


_retriever_instance = None

def get_retriever():
    """Get singleton retriever"""
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = RAGRetriever()
    return _retriever_instance