import pandas as pd
import numpy as np
import joblib
import networkx as nx
from pathlib import Path
from typing import Dict, List, Any, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
import sys
from pathlib import Path

# Add project root to Python path
root_dir = Path(__file__).parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.utils.logger import setup_logger
from src.utils.config import PROJECT_ROOT

logger = setup_logger(__name__)

class MobilityAnalyzer:
    """Analyze career mobility and skill similarity"""
    
    def __init__(self, model_dir: str = None):
        if model_dir is None:
            model_dir = PROJECT_ROOT / "4_models" / "mobility"
        
        self.model_dir = Path(model_dir)
        self.graph = None
        self.model = None
        self.embeddings = None
        self.load_model()
    
    def load_model(self):
        """Load graph, Node2Vec model, and embeddings"""
        try:
            # Load graph
            graph_path = self.model_dir / "graph.pkl"
            self.graph = joblib.load(graph_path)
            
            # Load Node2Vec model
            model_path = self.model_dir / "node2vec.model"
            self.model = Word2Vec.load(str(model_path))
            
            # Load embeddings
            emb_path = self.model_dir / "embeddings.pkl"
            self.embeddings = joblib.load(emb_path)
            
            logger.info("Mobility model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def find_similar_employees(
        self,
        employee_id: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Find similar employees based on skill embeddings
        
        Args:
            employee_id: Target employee ID
            top_k: Number of similar employees to return
        
        Returns:
            List of similar employees with similarity scores
        """
        if employee_id not in self.embeddings:
            logger.warning(f"Employee {employee_id} not found in embeddings")
            return []
        
        target_emb = self.embeddings[employee_id].reshape(1, -1)
        
        similarities = []
        
        for node, emb in self.embeddings.items():
            # Only compare with other employees
            if node.startswith('EMP') and node != employee_id:
                sim = cosine_similarity(target_emb, emb.reshape(1, -1))[0][0]
                similarities.append({
                    'employee_id': node,
                    'similarity_score': float(sim)
                })
        
        # Sort by similarity
        similarities.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        return similarities[:top_k]
    
    def get_employee_skills(self, employee_id: str) -> List[str]:
        """Get skills for an employee from the graph"""
        if not self.graph.has_node(employee_id):
            return []
        
        skills = []
        for neighbor in self.graph.neighbors(employee_id):
            if neighbor.startswith('skill_'):
                skill_name = neighbor.replace('skill_', '')
                skills.append(skill_name)
        
        return skills
    
    def recommend_career_paths(
        self,
        employee_id: str,
        target_role: str = None
    ) -> List[Dict[str, Any]]:
        """
        Recommend career paths based on similar employees
        
        Args:
            employee_id: Current employee
            target_role: Optional target role to filter by
        
        Returns:
            List of recommended paths with required skills
        """
        # Find similar employees
        similar_employees = self.find_similar_employees(employee_id, top_k=10)
        
        # Load employee data to get roles
        employees_path = PROJECT_ROOT / "data" / "employees.csv"
        employees_df = pd.read_csv(employees_path)
        
        # Get current employee's skills
        current_skills = set(self.get_employee_skills(employee_id))
        
        recommendations = []
        
        for similar in similar_employees:
            similar_id = similar['employee_id']
            
            # Get similar employee's info
            similar_info = employees_df[
                employees_df['employee_id'] == similar_id
            ]
            
            if len(similar_info) == 0:
                continue
            
            similar_role = similar_info.iloc[0]['job_role']
            similar_dept = similar_info.iloc[0]['department']
            
            # Skip if target role specified and doesn't match
            if target_role and similar_role != target_role:
                continue
            
            # Get similar employee's skills
            similar_skills = set(self.get_employee_skills(similar_id))
            
            # Calculate skill gap
            missing_skills = similar_skills - current_skills
            matched_skills = similar_skills & current_skills
            
            if len(similar_skills) > 0:
                match_percentage = len(matched_skills) / len(similar_skills) * 100
            else:
                match_percentage = 0
            
            recommendations.append({
                'target_role': similar_role,
                'department': similar_dept,
                'similarity_score': similar['similarity_score'],
                'skill_match_percentage': match_percentage,
                'missing_skills': list(missing_skills),
                'matched_skills': list(matched_skills),
                'required_skills_count': len(missing_skills)
            })
        
        # Remove duplicates and sort
        unique_recommendations = {}
        for rec in recommendations:
            key = (rec['target_role'], rec['department'])
            if key not in unique_recommendations or \
               rec['similarity_score'] > unique_recommendations[key]['similarity_score']:
                unique_recommendations[key] = rec
        
        final_recommendations = list(unique_recommendations.values())
        final_recommendations.sort(
            key=lambda x: (-x['skill_match_percentage'], x['required_skills_count'])
        )
        
        return final_recommendations[:5]
    
    def recommend_skills_for_role(
        self,
        target_role: str,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Recommend top skills for a specific role
        
        Args:
            target_role: Target job role
            top_k: Number of skills to recommend
        
        Returns:
            List of recommended skills with frequencies
        """
        # Load employee data
        employees_path = PROJECT_ROOT / "data" / "employees.csv"
        employees_df = pd.read_csv(employees_path)
        
        # Find employees in target role
        target_employees = employees_df[
            employees_df['job_role'] == target_role
        ]['employee_id'].tolist()
        
        # Count skill frequencies
        skill_counts = {}
        
        for emp_id in target_employees:
            skills = self.get_employee_skills(emp_id)
            for skill in skills:
                skill_counts[skill] = skill_counts.get(skill, 0) + 1
        
        # Sort by frequency
        sorted_skills = sorted(
            skill_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Calculate percentages
        total_employees = len(target_employees)
        
        recommendations = []
        for skill, count in sorted_skills[:top_k]:
            recommendations.append({
                'skill_name': skill,
                'frequency': count,
                'percentage': (count / total_employees * 100) if total_employees > 0 else 0,
                'importance': 'High' if count / total_employees > 0.7 else 'Medium' if count / total_employees > 0.4 else 'Low'
            })
        
        return recommendations
    
    def get_skill_graph_metrics(self) -> Dict[str, Any]:
        """Get overall graph metrics"""
        employee_nodes = [n for n in self.graph.nodes() if n.startswith('EMP')]
        skill_nodes = [n for n in self.graph.nodes() if n.startswith('skill_')]
        
        return {
            'total_employees': len(employee_nodes),
            'total_skills': len(skill_nodes),
            'total_connections': self.graph.number_of_edges(),
            'avg_skills_per_employee': self.graph.number_of_edges() / len(employee_nodes) if employee_nodes else 0,
            'graph_density': nx.density(self.graph)
        }

# Singleton instance
_analyzer_instance = None

def get_analyzer() -> MobilityAnalyzer:
    """Get or create singleton analyzer instance"""
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = MobilityAnalyzer()
    return _analyzer_instance