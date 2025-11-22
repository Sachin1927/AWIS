from typing import Dict, Any
from pathlib import Path
import sys
import traceback

CURRENT_FILE = Path(__file__).resolve()
SRC_DIR = CURRENT_FILE.parent.parent
PROJECT_ROOT = SRC_DIR.parent

sys.path.insert(0, str(SRC_DIR))

from utils.logger import setup_logger

logger = setup_logger(__name__)


class AWISAgent:
    """
    AI Chat Assistant for AWIS
    Completely rewritten for stability & no-freezing lazy-loading.
    """

    def __init__(self):
        logger.info("Initializing AWIS Agent (lazy-loading mode)...")

        # DO NOT LOAD ANY HEAVY MODELS HERE (prevents freezing)
        self.retriever = None
        self.predictor = None
        self.forecaster = None
        self.analyzer = None

        logger.info("AWIS Agent ready (models load only when needed)")

    # --------------------------------------------------------
    # LAZY LOADERS  (load models only when needed)
    # --------------------------------------------------------

    def _load_retriever(self):
        if self.retriever is None:
            try:
                logger.info("Loading RAG retriever...")
                from rag_index.retriever import get_retriever
                self.retriever = get_retriever()
                logger.info("RAG retriever loaded.")
            except Exception as e:
                logger.error(f"Retriever load failed: {e}")
                self.retriever = None
        return self.retriever

    def _load_predictor(self):
        if self.predictor is None:
            try:
                logger.info("Loading attrition predictor...")
                from ml.inference_attrition import get_predictor
                self.predictor = get_predictor()
                logger.info("Attrition predictor loaded.")
            except Exception as e:
                logger.error(f"Predictor load failed: {e}")
                self.predictor = None
        return self.predictor

    def _load_forecaster(self):
        if self.forecaster is None:
            try:
                logger.info("Loading skill forecaster...")
                from ml.inference_forecast import get_forecaster
                self.forecaster = get_forecaster()
                logger.info("Forecast model loaded.")
            except Exception as e:
                logger.error(f"Forecaster load failed: {e}")
                self.forecaster = None
        return self.forecaster

    def _load_analyzer(self):
        if self.analyzer is None:
            try:
                logger.info("Loading mobility analyzer...")
                from ml.inference_mobility import get_analyzer
                self.analyzer = get_analyzer()
                logger.info("Mobility analyzer loaded.")
            except Exception as e:
                logger.error(f"Analyzer load failed: {e}")
                self.analyzer = None
        return self.analyzer

    # --------------------------------------------------------
    # MAIN CHAT ROUTER
    # --------------------------------------------------------

    def chat(self, message: str) -> Dict[str, Any]:
        try:
            msg = message.lower().strip()

            if not msg:
                return self._handle_general(message)

            # Identify category
            if self._is_policy(msg):
                return self._handle_policy(message)

            if self._is_attrition(msg):
                return self._handle_attrition(message)

            if self._is_forecast(msg):
                return self._handle_forecast(message)

            if self._is_mobility(msg):
                return self._handle_mobility(message)

            return self._handle_general(message)

        except Exception as e:
            logger.error(traceback.format_exc())
            return {"response": f"Error: {str(e)}", "success": False}

    # --------------------------------------------------------
    # INTENT DETECTION
    # --------------------------------------------------------

    def _is_policy(self, msg):
        keys = ["policy", "remote", "promotion", "leave", "holiday", "work from home"]
        return any(k in msg for k in keys)

    def _is_attrition(self, msg):
        keys = ["attrition", "risk", "quit", "emp", "employee"]
        return any(k in msg for k in keys)

    def _is_forecast(self, msg):
        keys = ["forecast", "demand", "trend", "future", "skill"]
        return any(k in msg for k in keys)

    def _is_mobility(self, msg):
        keys = ["career", "path", "move", "transition", "role", "next"]
        return any(k in msg for k in keys)

    # --------------------------------------------------------
    # HANDLERS
    # --------------------------------------------------------

    def _handle_policy(self, message):
        retriever = self._load_retriever()

        if retriever is None:
            return {
                "response": "Policy search unavailable. Run the RAG index build.",
                "success": True
            }

        try:
            context = retriever.get_relevant_context(message, k=2)

            return {
                "response": f"📋 **HR Policy Info**\n\n{context['context'][:1200]}",
                "success": True
            }
        except Exception:
            return {"response": "Error searching policy documents.", "success": False}

    def _handle_attrition(self, message):
        predictor = self._load_predictor()

        if predictor is None:
            return {"response": "Attrition model unavailable.", "success": False}

        emp_id = self._extract_employee_id(message)

        if not emp_id:
            return {
                "response": "Please provide employee ID. Example: *Predict attrition for EMP1001*",
                "success": True
            }

        try:
            import pandas as pd
            df = pd.read_csv(PROJECT_ROOT / "data" / "employees.csv")

            if emp_id not in df["employee_id"].values:
                return {"response": f"Employee {emp_id} not found.", "success": False}

            row = df[df["employee_id"] == emp_id].iloc[0].to_dict()
            result = predictor.predict(row)

            return {
                "response": f"📊 **Attrition Risk for {emp_id}**\nRisk Level: {result['risk_level']}",
                "success": True
            }

        except Exception as e:
            return {"response": f"Error: {str(e)}", "success": False}

    def _handle_forecast(self, message):
        forecaster = self._load_forecaster()

        if forecaster is None:
            return {"response": "Forecast model unavailable.", "success": False}

        skill = None
        for s in ["python", "java", "sql", "aws", "react"]:
            if s in message.lower():
                skill = s.title()
                break

        if not skill:
            return {"response": "Specify a skill. Example: *Forecast Python demand*", "success": True}

        try:
            df = forecaster.forecast(skill, months_ahead=6)

            if df.empty:
                return {"response": f"No forecast available for {skill}.", "success": False}

            out = f"📈 **6 Month Forecast for {skill}**\n\n"
            for _, row in df.iterrows():
                out += f"{row['date']}: {row['forecasted_demand']} positions\n"

            return {"response": out, "success": True}

        except Exception as e:
            return {"response": f"Error: {str(e)}", "success": False}

    def _handle_mobility(self, message):
        analyzer = self._load_analyzer()

        if analyzer is None:
            return {"response": "Mobility model unavailable.", "success": False}

        emp_id = self._extract_employee_id(message)

        if not emp_id:
            return {"response": "Provide employee ID. Example: *Career path for EMP1005*", "success": True}

        try:
            skills = analyzer.get_employee_skills(emp_id)
            paths = analyzer.recommend_career_paths(emp_id)

            if not paths:
                return {"response": "No recommendations found.", "success": False}

            first = paths[0]

            out = f"🚀 **Career Path for {emp_id}**\nRole: {first['target_role']}\nMatch: {first['skill_match_percentage']:.1f}%"

            return {"response": out, "success": True}
        except Exception as e:
            return {"response": f"Error: {str(e)}", "success": False}

    def _handle_general(self, message):
        return {
            "response": "Ask about: HR Policy, Attrition, Forecast, Career Mobility.",
            "success": True
        }

    def _extract_employee_id(self, message):
        import re
        match = re.search(r"EMP\d+", message.upper())
        return match.group(0) if match else None

    def reset_memory(self):
        return


# GLOBAL SINGLETON
_agent_instance = None

def get_agent():
    global _agent_instance
    if _agent_instance is None:
        logger.info("Creating global AWIS Agent...")
        _agent_instance = AWISAgent()
        logger.info("AWIS Agent created.")
    return _agent_instance
