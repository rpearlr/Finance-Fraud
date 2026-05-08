import logging
from typing import Optional
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from agents.fraud_agent import FraudExpertAgent
from agents.rag_pipeline import DocumentAssistantAgent
from agents.investment_agent import InvestmentAdviceAgent

# Define a protocol or placeholder for the col function to be injected
_db_col_func = None

def set_db_col_func(func):
    global _db_col_func
    _db_col_func = func

log = logging.getLogger("orchestrator")

# Initialize agents once
_fraud_agent = FraudExpertAgent()
_rag_agent = DocumentAssistantAgent()
_investment_agent = InvestmentAdviceAgent()

@tool
def regulatory_query(question: str, source_filter: Optional[str] = None):
    """Answer questions about financial regulations (SEBI, RBI, Basel, FEMA, IRDAI, etc.) 
    using the document assistant RAG pipeline. 
    Use this for any questions about compliance, circulars, guidelines, or regulatory requirements.
    The source_filter can be 'SEBI', 'RBI', 'Basel', 'FEMA', or 'IRDAI' if specifically mentioned."""
    return _rag_agent.run(question, source_filter=source_filter)

@tool
def fraud_explanation(question: str):
    """Explain a specific fraud score or transaction pattern. 
    Use this if the user asks 'why' a transaction was scored as fraud, 
    or provides a transaction ID like 'TX-12345'. 
    The tool will automatically extract the transaction context if available."""
    # We'll handle the extraction here to make it autonomous
    from backend.utils import _extract_tx_context
    tx_id, fraud_score, features = _extract_tx_context(question)
    return _fraud_agent.explain(fraud_score=fraud_score, features=features, tx_id=tx_id)

@tool
def investment_advice(question: str):
    """Provide structured investment guidance, portfolio advice, or market insights.
    Use this for questions about asset allocation, mutual funds, stocks, or financial planning."""
    return _investment_agent.advise(question)

@tool
def database_stats(question: str):
    """Query transaction statistics, counts, and averages from the database.
    Use this for quantitative questions like 'how many transactions', 'average fraud score', 
    or 'total volume'."""
    if _db_col_func is None:
        return {"answer": "Database not initialized for orchestrator.", "sources": []}
    
    try:
        txs = _db_col_func("transactions")
        total = txs.count_documents({"fraud_score": {"$exists": True}})

        import pymongo
        agg = list(txs.aggregate([
            {"$match": {"fraud_score": {"$exists": True}}},
            {"$group": {
                "_id"      : None,
                "avg_score": {"$avg": "$fraud_score"},
                "max_score": {"$max": "$fraud_score"},
            }},
        ]))
        avg_score   = round(agg[0]["avg_score"], 4) if agg else 0
        max_score   = round(agg[0]["max_score"], 4) if agg else 0
        fraud_count = txs.count_documents({"label": 1})

        if total > 0:
            answer = (
                f"From the transactions database: {total:,} transactions have been scored. "
                f"Average fraud score: {avg_score:.4f}. "
                f"Highest score: {max_score:.4f}. "
                f"Flagged as fraud (score ≥ 0.5): {fraud_count:,} transactions "
                f"({(fraud_count / total * 100):.1f}% of total)."
            )
        else:
            answer = (
                "No scored transactions found in the database yet."
            )

        return {"answer": answer, "sources": ["MongoDB:transactions"], "chunks_used": 0, "tokens_used": 0}
    except Exception as e:
        return {"answer": f"Database query failed: {str(e)}", "sources": []}

class LangChainOrchestrator:
    def __init__(self, model_name="gpt-4o-mini"):
        from langchain_openai import AzureChatOpenAI
        import os
        
        self.llm = AzureChatOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
            azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o-mini"),
            temperature=0,
        )
        self.tools = [regulatory_query, fraud_explanation, investment_advice, database_stats]

    def run(self, question: str):
        """Execute the agent to answer the question."""
        log.info(f"Orchestrator running for: {question}")
        try:
            # Step 1: Decide which tool to call
            # We'll use a clear prompt to ensure the LLM chooses correctly
            messages = [
                ("system", "You are the FinRisk Orchestrator. Route the user's question to the most appropriate tool. "
                           "If no tool is suitable, answer the question yourself."),
                ("user", question)
            ]
            
            llm_with_tools = self.llm.bind_tools(self.tools)
            ai_msg = llm_with_tools.invoke(messages)
            
            answer = ai_msg.content
            agent_used = "langchain_orchestrator"
            extra_meta = {}
            
            # Step 2: Execute tool if the LLM requested it
            if ai_msg.tool_calls:
                tool_call = ai_msg.tool_calls[0]
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                
                log.info(f"Tool selected: {tool_name} with args: {tool_args}")
                
                # Map tool name to tool function
                tool_map = {t.name: t for t in self.tools}
                if tool_name in tool_map:
                    agent_used = tool_name
                    # Call the tool function
                    observation = tool_map[tool_name].invoke(tool_args)
                    
                    if isinstance(observation, dict):
                        answer = observation.get("answer") or observation.get("explanation") or str(observation)
                        extra_meta = {
                            "sources"    : observation.get("sources", []),
                            "chunks_used": observation.get("chunks_used", 0),
                            "tokens_used": observation.get("tokens_used", 0),
                            "fraud_score": observation.get("fraud_score"),
                            "risk_level" : observation.get("risk_level"),
                            "action"     : observation.get("action"),
                            "top_features": observation.get("top_features", []),
                            "thinking"   : observation.get("thinking", ""),
                        }
                    else:
                        answer = str(observation)
                else:
                    log.warning(f"Tool {tool_name} not found in tool map")
            
            # If the tool didn't provide a string answer (unlikely with our setup)
            if not answer:
                answer = "I've processed your request but don't have a specific answer. Try rephrasing your question."

            return {
                "answer": answer,
                "agent_used": agent_used,
                "extra_meta": extra_meta
            }
        except Exception as e:
            log.error(f"Orchestrator failed: {e}")
            import traceback
            log.error(traceback.format_exc())
            return {
                "answer": f"I encountered an error while processing your request: {str(e)}",
                "agent_used": "orchestrator_error",
                "extra_meta": {}
            }



    def route(self, question: str):
        """Deprecated: Use run() instead. Maintained for temporary compatibility."""
        return self.run(question)

