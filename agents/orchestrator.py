import logging
import os
from typing import Optional
from langchain_openai import AzureChatOpenAI
from langchain_core.tools import tool

from backend.database import query_db, query_one

log = logging.getLogger("orchestrator")

class LangChainOrchestrator:
    def __init__(self, fraud_agent, rag_agent, investment_agent):
        self.fraud_agent = fraud_agent
        self.rag_agent = rag_agent
        self.investment_agent = investment_agent

        self.llm = AzureChatOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
            azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o-mini"),
            temperature=0,
        )

        # Define tools as closures over the injected agent instances
        @tool
        def regulatory_query(question: str, source_filter: Optional[str] = None):
            """Answer questions about financial regulations (SEBI, RBI, Basel, FEMA, IRDAI, etc.) 
            using the document assistant RAG pipeline. 
            Use this for any questions about compliance, circulars, guidelines, or regulatory requirements.
            The source_filter can be 'SEBI', 'RBI', 'Basel', 'FEMA', or 'IRDAI' if specifically mentioned."""
            return self.rag_agent.run(question, source_filter=source_filter)

        @tool
        def fraud_explanation(question: str):
            """Explain a specific fraud score or transaction pattern. 
            Use this if the user asks 'why' a transaction was scored as fraud, 
            or provides a transaction ID like 'TX-12345'. 
            The tool will automatically extract the transaction context if available."""
            from backend.utils import _extract_tx_context
            tx_id, fraud_score, features = _extract_tx_context(question)
            return self.fraud_agent.explain(fraud_score=fraud_score, features=features, tx_id=tx_id)

        @tool
        def investment_advice(question: str):
            """Provide structured investment guidance, portfolio advice, or market insights.
            Use this for questions about asset allocation, mutual funds, stocks, or financial planning."""
            return self.investment_agent.advise(question)

        

        self.tools = [regulatory_query, fraud_explanation, investment_advice]

    def run(self, question: str):
        """Execute the agent to answer the question."""
        log.info(f"Orchestrator running for: {question}")
        try:
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
            
            if ai_msg.tool_calls:
                tool_call = ai_msg.tool_calls[0]
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                log.info(f"Tool selected: {tool_name} with args: {tool_args}")
                
                tool_map = {t.name: t for t in self.tools}
                if tool_name in tool_map:
                    agent_used = tool_name
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
