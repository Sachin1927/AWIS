from typing import List, Dict, Any
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

from src.utils.logger import setup_logger
from src.utils.config import PROJECT_ROOT, config

logger = setup_logger(__name__)

class RAGRetriever:
    """Retrieve relevant documents for RAG"""
    
    def __init__(
        self,
        vectorstore_path: str = None,
        embedding_model: str = None,
        k: int = 3
    ):
        if vectorstore_path is None:
            vectorstore_path = PROJECT_ROOT / "2_src" / "rag_index" / "vectorstore" / "faiss_index"
        
        if embedding_model is None:
            embedding_model = config['rag']['embeddings']['model']
        
        self.vectorstore_path = Path(vectorstore_path)
        self.k = k
        
        # Initialize embeddings
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': config['rag']['embeddings']['device']}
        )
        
        # Load vector store
        self.vectorstore = None
        self.load_vectorstore()
    
    def load_vectorstore(self):
        """Load FAISS vector store"""
        if not self.vectorstore_path.exists():
            logger.error(f"Vector store not found at {self.vectorstore_path}")
            logger.info("Please run: python 2_src/rag_index/build_index.py")
            raise FileNotFoundError(f"Vector store not found: {self.vectorstore_path}")
        
        logger.info(f"Loading vector store from {self.vectorstore_path}")
        self.vectorstore = FAISS.load_local(
            str(self.vectorstore_path),
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        logger.info("Vector store loaded successfully")
    
    def retrieve(
        self,
        query: str,
        k: int = None,
        filter_dict: Dict[str, Any] = None
    ) -> List[Document]:
        """
        Retrieve relevant documents
        
        Args:
            query: Search query
            k: Number of documents to retrieve
            filter_dict: Optional metadata filters
        
        Returns:
            List of relevant documents
        """
        if k is None:
            k = self.k
        
        logger.info(f"Retrieving {k} documents for query: {query[:100]}...")
        
        if filter_dict:
            docs = self.vectorstore.similarity_search(
                query,
                k=k,
                filter=filter_dict
            )
        else:
            docs = self.vectorstore.similarity_search(query, k=k)
        
        logger.info(f"Retrieved {len(docs)} documents")
        return docs
    
    def retrieve_with_scores(
        self,
        query: str,
        k: int = None
    ) -> List[tuple]:
        """Retrieve documents with similarity scores"""
        if k is None:
            k = self.k
        
        docs_and_scores = self.vectorstore.similarity_search_with_score(query, k=k)
        return docs_and_scores
    
    def format_docs_for_context(self, docs: List[Document]) -> str:
        """Format retrieved documents for LLM context"""
        context_parts = []
        
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get('source', 'Unknown')
            content = doc.page_content.strip()
            
            context_parts.append(
                f"[Source {i}: {source}]\n{content}\n"
            )
        
        return "\n".join(context_parts)
    
    def get_relevant_context(self, query: str, k: int = None) -> Dict[str, Any]:
        """
        Get relevant context with sources
        
        Returns:
            Dictionary with context and sources
        """
        docs = self.retrieve(query, k)
        
        context = self.format_docs_for_context(docs)
        sources = [doc.metadata.get('source', 'Unknown') for doc in docs]
        
        return {
            'context': context,
            'sources': sources,
            'num_docs': len(docs)
        }

# Singleton instance
_retriever_instance = None

def get_retriever() -> RAGRetriever:
    """Get or create singleton retriever instance"""
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = RAGRetriever()
    return _retriever_instance