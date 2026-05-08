"""
Investment Advice Agent — Phi-4 Mini Reasoning via Azure AI Foundry
Provides investment guidance based on market context and user queries.
"""

import os
import sys
import json
import logging
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
log = logging.getLogger("investment_agent")

# ── Config ─────────────────────────────────────────────────────────────────────
FOUNDRY_ENDPOINT   = 'https://finrisk-foundry.services.ai.azure.com/openai/v1'
FOUNDRY_API_KEY    = os.getenv("FOUNDRY_KEY")
PHI4_DEPLOYMENT    = "Phi-4"

# ══════════════════════════════════════════════════════════════════════════════
# PROMPT BUILDER
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are an Investment Advisor Agent at a financial institution.
Provide concise, responsible investment guidance.

Rules:
- Final advice: 120-160 words MAXIMUM.
- Structure: risk assessment (1 sentence) → 2-3 specific recommendations → disclaimer.
- Mandatory final line: "DISCLAIMER: Educational purposes only. Consult a certified financial planner before investing."
- Plain English. No excessive jargon.
- Never guarantee returns.

Knowledge: Indian markets (Sensex/Nifty/MF/PPF/FD), global asset classes, inflation/interest rate dynamics.
"""

def build_prompt(question: str) -> str:
    """Build the user prompt for the investment agent."""
    prompt = f"""Please provide investment guidance for the following question:
"{question}"

Remember to assess the likely risk profile implicit in the question, provide structured guidance, and include the mandatory disclaimer at the end.
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
        max_tokens=700,
        temperature=0.4,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", user_prompt),
    ])

    chain = prompt | llm | StrOutputParser()

    log.info(f"Calling Phi-4 Mini Reasoning via LangChain @ {FOUNDRY_ENDPOINT}")
    
    explanation = chain.invoke({})
    
    log.info(f"Phi-4 response — {len(explanation)} chars")

    return "", explanation, 0


# ══════════════════════════════════════════════════════════════════════════════
# INVESTMENT ADVICE AGENT
# ══════════════════════════════════════════════════════════════════════════════

class InvestmentAdviceAgent:
    """
    Investment Advice Agent.
    Takes a user question, provides structured advice using Phi-4.

    Usage in FastAPI /agent/query:
        from agents.investment_agent import InvestmentAdviceAgent
        investment_agent = InvestmentAdviceAgent()
        result = investment_agent.advise("What is a good asset allocation for high inflation?")
    """

    def __init__(self):
        self._check_config()

    def _check_config(self):
        missing = []
        if not FOUNDRY_ENDPOINT: missing.append("AZURE_FOUNDRY_ENDPOINT")
        if not FOUNDRY_API_KEY:  missing.append("AZURE_FOUNDRY_PHI4_KEY")
        if missing:
            log.warning(f"InvestmentAdviceAgent missing config: {missing}")

    def advise(self, question: str) -> dict:
        """
        Generate plain-English investment advice.

        Args:
            question: user's query

        Returns:
            dict containing explanation, latency, tokens used.
        """
        import time
        start = time.time()

        log.info(f"InvestmentAdviceAgent.advise — question='{question[:60]}'")

        try:
            user_prompt = build_prompt(question)
            thinking, explanation, tokens = call_phi4_langchain(SYSTEM_PROMPT, user_prompt)

            return {
                "explanation" : explanation,
                "thinking"    : thinking,
                "tokens_used" : tokens,
                "latency_ms"  : int((time.time() - start) * 1000),
                "model"       : PHI4_DEPLOYMENT,
                "agent"       : "investment_agent",
            }

        except Exception as e:
            log.error(f"InvestmentAdviceAgent error: {e}")
            return {
                "explanation": f"Agent error: {str(e)}",
                "thinking"   : "",
                "tokens_used": 0,
                "model"      : PHI4_DEPLOYMENT,
                "agent"      : "investment_agent",
            }

if __name__ == "__main__":
    agent = InvestmentAdviceAgent()
    res = agent.advise("I am 30 years old with a moderate risk appetite. Should I invest in crypto?")
    print("\n--- ADVICE ---\n")
    print(res["explanation"])
