import pandas as pd
import numpy as np
import networkx as nx
from node2vec import Node2Vec
from sklearn.metrics.pairwise import cosine_similarity
import joblib
from pathlib import Path
import sys


# Fix imports
CURRENT_FILE = Path(__file__).resolve()
CURRENT_DIR = CURRENT_FILE.parent  # src/ml/
SRC_DIR = CURRENT_DIR.parent  # src/
PROJECT_ROOT = SRC_DIR.parent  # AWIS/

sys.path.insert(0, str(SRC_DIR))

from utils.logger import setup_logger

logger = setup_logger(__name__)

class MobilityGraphTrainer:
    """Train employee mobility graph using Node2Vec"""
    
    def __init__(
        self,
        dimensions: int = 128,
        walk_length: int = 80,
        num_walks: int = 10,
        workers: int = 4
    ):
        self.dimensions = dimensions
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.workers = workers
        
        self.graph = None
        self.model = None
        self.embeddings = None
    
    def load_data(
        self,
        employees_path: str = None,
        skills_path: str = None
    ):
        """Load employee and skills data"""
        if employees_path is None:
            employees_path = PROJECT_ROOT / "data" / "employees.csv"
        if skills_path is None:
            skills_path = PROJECT_ROOT / "data" / "employee_skills.csv"
        
        logger.info("Loading data...")
        employees = pd.read_csv(employees_path)
        employee_skills = pd.read_csv(skills_path)
        
        logger.info(f"Employees: {len(employees)}, Skills mappings: {len(employee_skills)}")
        
        return employees, employee_skills
    
    def build_graph(
        self,
        employees: pd.DataFrame,
        employee_skills: pd.DataFrame
    ) -> nx.Graph:
        """Build bipartite graph: employees <-> skills"""
        logger.info("Building employee-skill graph...")
        
        G = nx.Graph()
        
        # Add employee nodes
        for _, emp in employees.iterrows():
            G.add_node(
                emp['employee_id'],
                node_type='employee',
                department=emp['department'],
                job_role=emp['job_role']
            )
        
        # Add skill nodes and edges
        for _, row in employee_skills.iterrows():
            skill_node = f"skill_{row['skill_name']}"
            
            # Add skill node if not exists
            if not G.has_node(skill_node):
                G.add_node(skill_node, node_type='skill')
            
            # Add edge with weight based on proficiency
            proficiency_weight = {
                'Beginner': 1,
                'Intermediate': 2,
                'Advanced': 3,
                'Expert': 4
            }.get(row['proficiency'], 2)
            
            G.add_edge(
                row['employee_id'],
                skill_node,
                weight=proficiency_weight
            )
        
        logger.info(f"Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        self.graph = G
        return G
    
    def train_node2vec(self):
        """Train Node2Vec model"""
        logger.info("Training Node2Vec (this may take a few minutes)...")
        
        # Initialize Node2Vec
        node2vec = Node2Vec(
            self.graph,
            dimensions=self.dimensions,
            walk_length=self.walk_length,
            num_walks=self.num_walks,
            workers=self.workers
        )
        
        # Train
        self.model = node2vec.fit(window=10, min_count=1, batch_words=4)
        
        # Extract embeddings
        self.embeddings = {}
        for node in self.graph.nodes():
            self.embeddings[node] = self.model.wv[node]
        
        logger.info(f"Node2Vec training complete. Embedding dim: {self.dimensions}")
        
        return self.model
    
    def save_model(self, output_dir: str = None):
        """Save graph and model"""
        if output_dir is None:
            output_dir = PROJECT_ROOT / "4_models" / "mobility"
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save graph
        graph_path = output_dir / "graph.pkl"
        joblib.dump(self.graph, graph_path)
        logger.info(f"\nGraph saved to {graph_path}")
        
        # Save Node2Vec model
        model_path = output_dir / "node2vec.model"
        self.model.save(str(model_path))
        logger.info(f"Node2Vec model saved to {model_path}")
        
        # Save embeddings
        emb_path = output_dir / "embeddings.pkl"
        joblib.dump(self.embeddings, emb_path)
        logger.info(f"Embeddings saved to {emb_path}")


if __name__ == "__main__":
    logger.info("="*60)
    logger.info("MOBILITY GRAPH TRAINING")
    logger.info("="*60)
    logger.info("")
    
    trainer = MobilityGraphTrainer()
    employees, employee_skills = trainer.load_data()
    trainer.build_graph(employees, employee_skills)
    trainer.train_node2vec()
    trainer.save_model()
    
    logger.info("")
    logger.info("="*60)
    logger.info("✅ MOBILITY GRAPH TRAINING COMPLETE!")
    logger.info("="*60)