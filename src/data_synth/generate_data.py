import pandas as pd
import numpy as np
from faker import Faker
from pathlib import Path
import yaml
from typing import Dict, List, Any
import random
from datetime import datetime, timedelta

# Setup
fake = Faker()
Faker.seed(42)
np.random.seed(42)
random.seed(42)

PROJECT_ROOT = Path(__file__).parent.parent.parent

class SyntheticDataGenerator:
    """Generate synthetic HR data"""
    
    def __init__(self, schema_path: str = None):
        if schema_path is None:
            schema_path = PROJECT_ROOT / "src" / "data_synth" / "schema.yaml"
        
        with open(schema_path, 'r') as f:
            self.schema = yaml.safe_load(f)
        
        self.employees_df = None
        self.skills_df = None
        self.employee_skills_df = None
        self.historical_data_df = None
    
    def generate_employees(self) -> pd.DataFrame:
        """Generate employee data"""
        count = self.schema['employees']['count']
        fields = self.schema['employees']['fields']
        
        data = []
        
        for i in range(count):
            employee = {}
            
            # Employee ID
            employee['employee_id'] = f"EMP{1000 + i}"
            
            # Personal info
            employee['name'] = fake.name()
            employee['email'] = fake.email()
            employee['age'] = np.random.randint(22, 66)
            
            # Job info
            employee['department'] = np.random.choice(
                fields['department']['choices']
            )
            employee['job_role'] = np.random.choice(
                fields['job_role']['choices']
            )
            
            # Experience
            employee['years_at_company'] = np.random.randint(0, 31)
            employee['last_promotion_years'] = np.random.randint(
                0, min(employee['years_at_company'] + 1, 11)
            )
            
            # Performance
            employee['performance_rating'] = np.random.choice(
                fields['performance_rating']['choices'],
                p=fields['performance_rating']['weights']
            )
            employee['satisfaction_level'] = np.random.uniform(0.0, 1.0)
            employee['training_hours'] = np.random.randint(0, 101)
            
            # Compensation
            base_salary = np.random.randint(30000, 100000)
            experience_bonus = employee['years_at_company'] * 2000
            performance_bonus = employee['performance_rating'] * 5000
            employee['monthly_income'] = int(
                base_salary + experience_bonus + performance_bonus
            )
            
            # Attrition factors
            # Low satisfaction + low performance + no promotion = higher attrition
            attrition_score = (
                (1 - employee['satisfaction_level']) * 0.4 +
                (5 - employee['performance_rating']) / 5 * 0.3 +
                min(employee['last_promotion_years'] / 10, 1) * 0.3
            )
            employee['attrition'] = 1 if attrition_score > 0.6 else 0
            
            data.append(employee)
        
        self.employees_df = pd.DataFrame(data)
        return self.employees_df
    
    def generate_skills(self) -> pd.DataFrame:
        """Generate skills catalog"""
        skills_data = []
        skill_id = 1
        
        for category, skills in self.schema['skills']['categories'].items():
            for skill_name in skills:
                skills_data.append({
                    'skill_id': f"SKL{skill_id:03d}",
                    'skill_name': skill_name,
                    'category': category
                })
                skill_id += 1
        
        self.skills_df = pd.DataFrame(skills_data)
        return self.skills_df
    
    def generate_employee_skills(self) -> pd.DataFrame:
        """Generate employee-skills mapping"""
        if self.employees_df is None:
            self.generate_employees()
        if self.skills_df is None:
            self.generate_skills()
        
        config = self.schema['employee_skills']
        proficiency_levels = config['proficiency_levels']
        
        data = []
        
        for _, employee in self.employees_df.iterrows():
            # Number of skills per employee
            num_skills = np.random.randint(
                config['skills_per_employee']['min'],
                config['skills_per_employee']['max'] + 1
            )
            
            # Randomly select skills
            selected_skills = self.skills_df.sample(n=num_skills)
            
            for _, skill in selected_skills.iterrows():
                data.append({
                    'employee_id': employee['employee_id'],
                    'skill_id': skill['skill_id'],
                    'skill_name': skill['skill_name'],
                    'proficiency': np.random.choice(proficiency_levels),
                    'years_experience': np.random.randint(
                        0, min(employee['years_at_company'] + 1, 16)
                    ),
                    'last_used_date': fake.date_between(
                        start_date='-2y',
                        end_date='today'
                    ).strftime('%Y-%m-%d')
                })
        
        self.employee_skills_df = pd.DataFrame(data)
        return self.employee_skills_df
    
    def generate_historical_skill_demand(
        self,
        months: int = 36
    ) -> pd.DataFrame:
        """Generate historical skill demand data for forecasting"""
        if self.skills_df is None:
            self.generate_skills()
        
        # Get top 20 skills to track
        top_skills = self.skills_df.head(20)
        
        # Generate monthly data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=months * 30)
        
        date_range = pd.date_range(start=start_date, end=end_date, freq='M')
        
        data = []
        
        for skill_idx, skill in top_skills.iterrows():
            # Create a trend for each skill
            base_demand = np.random.randint(10, 100)
            trend = np.random.uniform(-0.5, 2.0)  # Growth or decline
            seasonality = np.random.uniform(0, 10)
            
            for month_idx, date in enumerate(date_range):
                # Trend + Seasonality + Noise
                demand = (
                    base_demand +
                    trend * month_idx +
                    seasonality * np.sin(2 * np.pi * month_idx / 12) +
                    np.random.normal(0, 5)
                )
                demand = max(0, int(demand))
                
                data.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'skill_id': skill['skill_id'],
                    'skill_name': skill['skill_name'],
                    'demand_count': demand,
                    'job_postings': int(demand * np.random.uniform(0.8, 1.2)),
                    'avg_salary': np.random.randint(50000, 150000)
                })
        
        self.historical_data_df = pd.DataFrame(data)
        return self.historical_data_df
    
    def save_all(self, output_dir: str = None):
        """Generate and save all datasets"""
        if output_dir is None:
            output_dir = PROJECT_ROOT / "data"
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate all
        print("Generating employees...")
        self.generate_employees()
        
        print("Generating skills...")
        self.generate_skills()
        
        print("Generating employee-skills mapping...")
        self.generate_employee_skills()
        
        print("Generating historical skill demand...")
        self.generate_historical_skill_demand()
        
        # Save
        print(f"\nSaving to {output_dir}...")
        self.employees_df.to_csv(output_dir / "employees.csv", index=False)
        self.skills_df.to_csv(output_dir / "skills.csv", index=False)
        self.employee_skills_df.to_csv(output_dir / "employee_skills.csv", index=False)
        self.historical_data_df.to_csv(output_dir / "historical_skill_demand.csv", index=False)
        
        print("\n✅ Data generation complete!")
        print(f"  - Employees: {len(self.employees_df)}")
        print(f"  - Skills: {len(self.skills_df)}")
        print(f"  - Employee-Skills: {len(self.employee_skills_df)}")
        print(f"  - Historical records: {len(self.historical_data_df)}")

if __name__ == "__main__":
    generator = SyntheticDataGenerator()
    generator.save_all()