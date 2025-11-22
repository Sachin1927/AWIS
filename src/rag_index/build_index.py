import os
from pathlib import Path
from typing import List
import sys

CURRENT_FILE = Path(__file__).resolve()
SRC_DIR = CURRENT_FILE.parent.parent
PROJECT_ROOT = SRC_DIR.parent

sys.path.insert(0, str(SRC_DIR))

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from utils.logger import setup_logger

logger = setup_logger(__name__)

class RAGIndexBuilder:
    """Build vector store index for RAG"""
    
    def __init__(
        self,
        docs_dir=None,
        vectorstore_dir=None,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2"
    ):
        if docs_dir is None:
            docs_dir = SRC_DIR / "rag_index" / "docs"
        if vectorstore_dir is None:
            vectorstore_dir = SRC_DIR / "rag_index" / "vectorstore"
        
        self.docs_dir = Path(docs_dir)
        self.vectorstore_dir = Path(vectorstore_dir)
        self.embedding_model_name = embedding_model
        
        # Create directories
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        self.vectorstore_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize embeddings
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'}
        )
        
        self.vectorstore = None
    
    def create_sample_documents(self):
        """Create sample HR policy documents"""
        logger.info("Creating sample HR policy documents...")
        
        sample_docs = {
            "remote_work_policy.txt": """
# Remote Work Policy

## Eligibility
All employees who have completed their probationary period (90 days) are eligible for remote work.

## Work Schedule
- Employees may work remotely up to 3 days per week
- Core hours: 10 AM - 3 PM (must be available)
- Flexible hours outside core time

## Equipment
- Company provides laptop and necessary software
- Employees responsible for internet connection
- IT support available 24/7

## Performance
- Same productivity standards as in-office work
- Regular check-ins with manager (weekly minimum)
""",
            "learning_development.txt": """
# Learning & Development Policy

## Training Budget
Each employee receives an annual professional development budget:
- Individual Contributors: $2,000/year
- Managers: $3,000/year
- Senior Leaders: $5,000/year

## Eligible Expenses
- Online courses and certifications
- Conference attendance
- Books and learning materials
- Professional coaching

## Time Off for Learning
- Up to 40 hours/year for skill development
- Additional time for relevant certifications

## Certification Incentives
- Company covers exam fees
- Bonus upon completion: $500-$2,000
""",
            "performance_review.txt": """
# Performance Review Process

## Review Cycle
- Annual reviews: December
- Mid-year check-ins: June
- Quarterly 1-on-1s with manager

## Performance Ratings
1. Needs Improvement
2. Meets Expectations
3. Exceeds Expectations
4. Outstanding
5. Exceptional

## Components
- Goal achievement (40%)
- Competencies (30%)
- Leadership/collaboration (20%)
- Innovation/initiative (10%)
""",
            "promotion_criteria.txt": """
# Promotion Criteria

## General Requirements
- Minimum 12 months in current role
- Meets or exceeds performance expectations
- Demonstrates readiness for next level

## Levels
### Individual Contributor
- IC1: Entry (0-2 years)
- IC2: Intermediate (2-4 years)
- IC3: Senior (4-7 years)
- IC4: Staff (7-10 years)

### Management
- M1: Team Lead
- M2: Manager
- M3: Senior Manager
- M4: Director

## Process
1. Self-nomination or manager recommendation
2. Promotion packet preparation
3. Peer review
4. Committee review
"""
        }
        
        for filename, content in sample_docs.items():
            filepath = self.docs_dir / filename
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            logger.info(f"Created: {filename}")
    
    def load_documents(self):
        """Load documents from docs directory"""
        logger.info(f"Loading documents from {self.docs_dir}")
        
        documents = []
        
        # Load text files
        for txt_file in self.docs_dir.glob("*.txt"):
            try:
                loader = TextLoader(str(txt_file), encoding='utf-8')
                documents.extend(loader.load())
            except Exception as e:
                logger.error(f"Error loading {txt_file}: {e}")
        
        # Load PDFs
        for pdf_file in self.docs_dir.glob("*.pdf"):
            try:
                loader = PyPDFLoader(str(pdf_file))
                documents.extend(loader.load())
            except Exception as e:
                logger.error(f"Error loading {pdf_file}: {e}")
        
        logger.info(f"Loaded {len(documents)} documents")
        
        if len(documents) == 0:
            logger.warning("No documents found. Creating sample documents...")
            self.create_sample_documents()
            return self.load_documents()
        
        return documents
    
    def split_documents(self, documents, chunk_size=1000, chunk_overlap=200):
        """Split documents into chunks"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        
        chunks = text_splitter.split_documents(documents)
        logger.info(f"Created {len(chunks)} chunks")
        return chunks
    
    def build_index(self):
        """Build vector store index"""
        documents = self.load_documents()
        chunks = self.split_documents(documents)
        
        logger.info("Building ChromaDB index...")
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=str(self.vectorstore_dir)
        )
        
        logger.info("Index built successfully")
        return self.vectorstore
    
    def load_index(self):
        """Load existing vector store"""
        logger.info(f"Loading vector store from {self.vectorstore_dir}")
        
        if not (self.vectorstore_dir / "chroma.sqlite3").exists():
            logger.warning("No existing index. Building new one...")
            return self.build_index()
        
        self.vectorstore = Chroma(
            persist_directory=str(self.vectorstore_dir),
            embedding_function=self.embeddings
        )
        
        logger.info("Vector store loaded")
        return self.vectorstore


if __name__ == "__main__":
    builder = RAGIndexBuilder()
    builder.build_index()
    logger.info("\n✅ RAG index built successfully!")