"""
Fraud ML Expert Agent — Phi-4 Mini Reasoning via Azure AI Foundry
Explains fraud predictions in plain English using chain-of-thought reasoning.

Run standalone:
  python agents/fraud_agent.py --score 0.94 --tx-id TX-94821

Import in FastAPI:
  from agents.fraud_agent import FraudExpertAgent
  agent = FraudExpertAgent()
  result = agent.explain(fraud_score, features, tx_id)
"""

import os
import sys
import json
import pickle
import logging
import argparse
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("fraud_agent")

# ── Config ─────────────────────────────────────────────────────────────────────
FOUNDRY_ENDPOINT   = 'https://finrisk-foundry.services.ai.azure.com/openai/v1'
FOUNDRY_API_KEY    = os.getenv("FOUNDRY_KEY")
PHI4_DEPLOYMENT    = "Phi-4"

MODEL_PATH         = Path(r"C:\Users\User\Desktop\fin-prj\ml\models\fraud_model.pkl")
FEATURES_PATH      = Path(r"C:\Users\User\Desktop\fin-prj\ml\models\feature_names.txt")

# Feature descriptions for the LLM — maps column name to human-readable meaning
FEATURE_DESCRIPTIONS = {
    "amount_zscore" : "transaction amount z-score (how many std deviations from mean; >2 = unusually large)",
    "velocity_24tx" : "rolling sum of last 24 transaction amounts in INR (high = burst activity)",
    "hour_of_day"   : "hour the transaction occurred (0-23; 0-5 = late night, high-risk window)",
    "amount_log"    : "log-scaled transaction amount (log1p)",
    "V1"            : "PCA component 1 (card behaviour pattern)",
    "V2"            : "PCA component 2 (merchant interaction pattern)",
    "V3"            : "PCA component 3 (transaction frequency pattern)",
    "V4"            : "PCA component 4 (geograpFhic pattern)",
    "V5"            : "PCA component 5 (device fingerprint pattern)",
    "V6"            : "PCA component 6",
    "V7"            : "PCA component 7",
    "V8"            : "PCA component 8",
    "V9"            : "PCA component 9",
    "V10"           : "PCA component 10",
    "V11"           : "PCA component 11",
    "V12"           : "PCA component 12",
    "V13"           : "PCA component 13",
    "V14"           : "PCA component 14 (strongly correlated with fraud in research)",
    "V15"           : "PCA component 15",
    "V16"           : "PCA component 16",
    "V17"           : "PCA component 17 (strongly correlated with fraud in research)",
    "V18"           : "PCA component 18",
    "V19"           : "PCA component 19",
    "V20"           : "PCA component 20",
    "V21"           : "PCA component 21",
    "V22"           : "PCA component 22",
    "V23"           : "PCA component 23",
    "V24"           : "PCA component 24",
    "V25"           : "PCA component 25",
    "V26"           : "PCA component 26",
    "V27"           : "PCA component 27",
    "V28"           : "PCA component 28",
}




# ══════════════════════════════════════════════════════════════════════════════
# FEATURE IMPORTANCE — get top contributors from XGBoost model
# ══════════════════════════════════════════════════════════════════════════════

def get_top_features(features: dict, top_n: int = 8) -> list[dict]:
    """
    Load the XGBoost model and extract the top-n most important features,
    weighted by both model importance and the actual feature values.
    Returns a ranked list of dicts with name, value, importance, description.
    """
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)

        feature_names = FEATURES_PATH.read_text(encoding="utf-8").strip().splitlines()
        importances   = model.feature_importances_

        ranked = []
        for name, importance in zip(feature_names, importances):
            value = features.get(name, 0.0)
            ranked.append({
                "name"       : name,
                "value"      : round(float(value), 4),
                "importance" : round(float(importance), 4),
                "description": FEATURE_DESCRIPTIONS.get(name, "PCA component"),
            })

        # Sort by model importance descending
        ranked.sort(key=lambda x: x["importance"], reverse=True)
        return ranked[:top_n]

    except FileNotFoundError:
        log.warning("Model file not found — using raw feature values only")
        # Fall back to engineered features only
        priority = ["amount_zscore", "velocity_24tx", "hour_of_day", "amount_log",
                    "V14", "V17", "V12", "V10"]
        return [
            {
                "name"       : k,
                "value"      : round(float(features.get(k, 0.0)), 4),
                "importance" : 0.0,
                "description": FEATURE_DESCRIPTIONS.get(k, "feature"),
            }
            for k in priority if k in features
        ]


def get_risk_level(score: float) -> tuple[str, str, str]:
    """
    Categorize fraud score into risk levels.
    Boundaries: High >= 0.80, Medium >= 0.50, else Low.
    """
    if score >= 0.80:
        return "HIGH", "🔴", "Immediate review required"
    elif score >= 0.50:
        return "MEDIUM", "🟡", "Flag for manual review"
    else:
        return "LOW", "🟢", "Within normal parameters"


# ══════════════════════════════════════════════════════════════════════════════
# PROMPT BUILDER
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are a fraud investigation expert at a major Indian financial institution.
You explain machine learning fraud detection decisions to compliance officers.

Rules:
- Final explanation: 80-120 words MAXIMUM. Investigators are busy.
- Reference the actual feature values given (never invent numbers).
- Structure: one-sentence verdict → 2-3 bullet reasons → one recommended action.
- Plain English, no jargon.
"""

def build_prompt(
    fraud_score : float,
    risk_level  : str,
    action      : str,
    top_features: list[dict],
    tx_id       : Optional[str],
    extra_context: Optional[dict],
) -> str:
    """Build the user prompt with full transaction context for Phi-4."""

    # Format top features as a readable table
    feature_lines = []
    for f in top_features:
        bar = "#" * int(f["importance"] * 40) if f["importance"] > 0 else "-"
        feature_lines.append(
            f"  {f['name']:<20} value={f['value']:>8.4f}  "
            f"importance={f['importance']:.4f}  [{bar}]"
        )
    feature_table = "\n".join(feature_lines)

    # Optional extra context (merchant, location, etc.)
    context_block = ""
    if extra_context:
        context_lines = [f"  {k}: {v}" for k, v in extra_context.items()]
        context_block = "\nAdditional context:\n" + "\n".join(context_lines)

    prompt = f"""A transaction has been scored by our XGBoost fraud classifier.

Transaction ID : {tx_id or 'N/A'}
Fraud Score    : {fraud_score:.4f} ({fraud_score*100:.1f}% probability of fraud)
Risk Level     : {risk_level}
Recommended    : {action}
{context_block}

Top contributing features (ranked by model importance):
{feature_table}

Please explain:
1. Why this transaction received this fraud score
2. Which specific feature values are most suspicious and why
3. What pattern of fraud this most likely represents (if high risk)
4. What the investigator should look for when reviewing this case

Be specific about the feature values — reference the actual numbers above.
"""
    return prompt


# ══════════════════════════════════════════════════════════════════════════════
# PHI-4 MINI REASONING CLIENT
# ══════════════════════════════════════════════════════════════════════════════

def call_phi4_langchain(system_prompt: str, user_prompt: str) -> tuple[str, str, int]:
    """
    Call Phi-4 Mini Reasoning via Azure AI Foundry using LangChain.
    Returns (thinking, explanation, tokens).
    """
    if not FOUNDRY_ENDPOINT or not FOUNDRY_API_KEY:
        raise ValueError(
            "AZURE_FOUNDRY_ENDPOINT or AZURE_FOUNDRY_PHI4_KEY not set in .env"
        )

    llm = ChatOpenAI(
        base_url=FOUNDRY_ENDPOINT,
        api_key=FOUNDRY_API_KEY,
        model=PHI4_DEPLOYMENT,
        max_tokens=500,
        temperature=0.4,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", user_prompt),
    ])

    chain = prompt | llm | StrOutputParser()

    log.info(f"Calling Phi-4 via LangChain @ {FOUNDRY_ENDPOINT}")
    
    # LangChain doesn't easily expose the raw response for tokens in a simple pipe without extra effort, 
    # but we can use callbacks or just accept that we might lose token counts for now or use invoke with config.
    # For simplicity and to stick to standard LCEL:
    explanation = chain.invoke({})
    
    log.info(f"Phi-4 response — {len(explanation)} chars")

    return "", explanation, 0 # Token count set to 0 for now as it's harder to get from simple invoke


# ══════════════════════════════════════════════════════════════════════════════
# FRAUD EXPERT AGENT
# ══════════════════════════════════════════════════════════════════════════════

class FraudExpertAgent:
    """
    ML Expert Agent for fraud explanation.
    Takes a fraud score + feature dict, returns a plain-English explanation
    generated by Phi-4 Mini Reasoning.

    Usage in FastAPI /agent/query:
        from agents.fraud_agent import FraudExpertAgent
        fraud_agent = FraudExpertAgent()
        result = fraud_agent.explain(
            fraud_score=0.94,
            features={"amount_zscore": 4.1, "velocity_24tx": 8200, ...},
            tx_id="TX-94821"
        )
    """

    def __init__(self):
        self._check_config()

    def _check_config(self):
        missing = []
        if not FOUNDRY_ENDPOINT: missing.append("AZURE_FOUNDRY_ENDPOINT")
        if not FOUNDRY_API_KEY:  missing.append("AZURE_FOUNDRY_PHI4_KEY")
        if missing:
            log.warning(f"FraudExpertAgent missing config: {missing}")

    def explain(
        self,
        fraud_score  : float,
        features     : dict,
        tx_id        : Optional[str] = None,
        extra_context: Optional[dict] = None,
    ) -> dict:
        """
        Generate a plain-English explanation for a fraud prediction.

        Args:
            fraud_score   : float 0-1 from XGBoost /predict endpoint
            features      : dict of feature_name -> value (same as /predict input)
            tx_id         : optional transaction ID for logging
            extra_context : optional dict of extra info (merchant, location, etc.)

        Returns:
            explanation   : str   — plain-English explanation for investigators
            thinking      : str   — Phi-4 chain-of-thought reasoning (debug use)
            risk_level    : str   — HIGH / MEDIUM / LOW
            action        : str   — recommended action
            top_features  : list  — ranked feature contributions
            fraud_score   : float — echo of input score
            tokens_used   : int   — Phi-4 tokens consumed
            agent         : str   — "fraud_expert_agent"
        """
        import time
        start = time.time()

        log.info(f"FraudExpertAgent.explain — tx_id={tx_id} score={fraud_score:.4f}")

        try:
            risk_level, icon, action = get_risk_level(fraud_score)
            top_features = get_top_features(features, top_n=8)

            user_prompt = build_prompt(
                fraud_score=fraud_score,
                risk_level=f"{icon} {risk_level}",
                action=action,
                top_features=top_features,
                tx_id=tx_id,
                extra_context=extra_context,
            )

            thinking, explanation, tokens = call_phi4_langchain(SYSTEM_PROMPT, user_prompt)

            return {
                "tx_id"       : tx_id,
                "fraud_score" : fraud_score,
                "risk_level"  : risk_level,
                "action"      : action,
                "explanation" : explanation,
                "thinking"    : thinking,   # chain-of-thought from Phi-4
                "top_features": top_features,
                "tokens_used" : tokens,
                "latency_ms"  : int((time.time() - start) * 1000),
                "model"       : PHI4_DEPLOYMENT,
                "agent"       : "fraud_expert_agent",
            }

        except Exception as e:
            log.error(f"FraudExpertAgent error: {e}")
            return {
                "tx_id"      : tx_id,
                "fraud_score": fraud_score,
                "risk_level" : "UNKNOWN",
                "action"     : "Manual review required",
                "explanation": f"Agent error: {str(e)}",
                "thinking"   : "",
                "top_features": [],
                "tokens_used": 0,
                "model" : "Phi-4" ,
                "agent"      : "fraud_expert_agent",
            }

    def explain_from_predict_response(self, predict_response: dict) -> dict:
        """
        Convenience wrapper — pass the raw /predict API response directly.

        predict_response: the JSON returned by POST /predict
        Example:
            pred = requests.post("/predict", json=features).json()
            explanation = fraud_agent.explain_from_predict_response(pred)
        """
        return self.explain(
            fraud_score=predict_response.get("fraud_score", 0.0),
            features=predict_response.get("features", {}),
            tx_id=predict_response.get("tx_id"),
        )


# ══════════════════════════════════════════════════════════════════════════════
# CLI — for standalone testing
# ══════════════════════════════════════════════════════════════════════════════

def make_sample_features(fraud_score: float) -> dict:
    """
    Generate realistic sample features for CLI testing.
    High score -> suspicious feature values. Low score -> normal values.
    """
    import random
    random.seed(42)

    if fraud_score >= 0.80:
        # High fraud — unusual values
        return {
            "V1": -3.2, "V2": 2.8, "V3": -2.1, "V4": 1.5, "V5": -3.8,
            "V6": 0.3,  "V7": -2.9,"V8": 1.1,  "V9": -0.8,"V10": -3.1,
            "V11": 1.4, "V12":-3.6,"V13": 0.2, "V14":-4.1,"V15": 0.6,
            "V16":-1.8, "V17":-3.4,"V18": 0.4, "V19": 0.1,"V20": 0.3,
            "V21": 0.7, "V22":-0.2,"V23": 0.1, "V24":-0.4,"V25": 0.2,
            "V26": 0.1, "V27": 0.5,"V28": 0.2,
            "amount_zscore": 4.1,
            "velocity_24tx": 8200.0,
            "hour_of_day"  : 3,
            "amount_log"   : 9.2,
        }
    elif fraud_score >= 0.50:
        # Medium fraud — some suspicious values
        return {
            "V1": -1.2, "V2": 0.8, "V3": -0.9, "V4": 0.5, "V5": -1.3,
            "V6": 0.2,  "V7": -0.9,"V8": 0.4,  "V9": -0.3,"V10": -1.1,
            "V11": 0.4, "V12":-1.2,"V13": 0.1, "V14":-1.8,"V15": 0.2,
            "V16":-0.6, "V17":-1.1,"V18": 0.1, "V19": 0.0,"V20": 0.1,
            "V21": 0.2, "V22":-0.1,"V23": 0.0, "V24":-0.1,"V25": 0.1,
            "V26": 0.0, "V27": 0.1,"V28": 0.1,
            "amount_zscore": 1.9,
            "velocity_24tx": 3400.0,
            "hour_of_day"  : 23,
            "amount_log"   : 7.1,
        }
    else:
        # Low fraud — normal values
        return {
            "V1": 0.1,  "V2": 0.2, "V3": -0.1, "V4": 0.1, "V5": -0.2,
            "V6": 0.1,  "V7": 0.0, "V8": 0.1,  "V9": 0.0, "V10": 0.1,
            "V11": 0.1, "V12": 0.0,"V13": 0.1, "V14": 0.0,"V15": 0.1,
            "V16": 0.0, "V17": 0.1,"V18": 0.0, "V19": 0.0,"V20": 0.0,
            "V21": 0.1, "V22": 0.0,"V23": 0.0, "V24": 0.0,"V25": 0.0,
            "V26": 0.0, "V27": 0.0,"V28": 0.0,
            "amount_zscore": 0.3,
            "velocity_24tx": 850.0,
            "hour_of_day"  : 14,
            "amount_log"   : 5.2,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fraud Expert Agent CLI")
    parser.add_argument("--score",  type=float, default=0.94,
                        help="Fraud score 0-1 (default: 0.94)")
    parser.add_argument("--tx-id", type=str,   default="TX-94821",
                        help="Transaction ID for display")
    parser.add_argument("--show-thinking", action="store_true",
                        help="Print Phi-4 chain-of-thought reasoning")
    parser.add_argument("--features-json", type=str, default=None,
                        help="JSON string of feature values (optional)")
    args = parser.parse_args()

    # Features: use provided JSON or generate sample based on score
    if args.features_json:
        try:
            features = json.loads(args.features_json)
        except json.JSONDecodeError as e:
            print(f"Invalid features JSON: {e}")
            sys.exit(1)
    else:
        features = make_sample_features(args.score)
        log.info("Using auto-generated sample features based on score")

    agent  = FraudExpertAgent()
    result = agent.explain(
        fraud_score=args.score,
        features=features,
        tx_id=args.tx_id,
        extra_context={"merchant_category": "Online Electronics", "location": "Mumbai"},
    )

    print("\n" + "=" * 65)
    print("FRAUD EXPERT AGENT — EXPLANATION REPORT")
    print("=" * 65)
    print(f"Transaction : {result['tx_id']}")
    print(f"Fraud Score : {result['fraud_score']:.4f} ({result['fraud_score']*100:.1f}%)")
    print(f"Risk Level  : {result['risk_level']}")
    print(f"Action      : {result['action']}")
    print(f"Model       : {result['model']}")

    if args.show_thinking and result["thinking"]:
        print("\n--- PHI-4 REASONING (chain-of-thought) ---")
        print(result["thinking"])

    print("\n--- EXPLANATION ---")
    print(result["explanation"])

    print("\n--- TOP CONTRIBUTING FEATURES ---")
    for f in result["top_features"]:
        bar = "#" * int(f["importance"] * 40)
        print(f"  {f['name']:<20} {f['value']:>8.4f}  [{bar}]")

    print("=" * 65)