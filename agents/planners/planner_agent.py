import json
from orchestration.llm_provider import LLMProvider
from orchestration.execution.agent_executor import AgentExecutor
SYSTEM_PROMPT = """
You are the Planner Agent of an autonomous AI system.

Your responsibility:
- Understand user intent
- Classify the task
- Decide which agents are needed
- Decide which tools are needed
- Estimate task complexity

AVAILABLE AGENTS:
- retrieval_agent
- sql_agent
- critic_agent
- synthesis_agent

AVAILABLE TOOLS:
- filesystem
- vector_search
- sql_database
- web_search
- python_executor

QUERY TYPES:
- retrieval
- sql
- file_analysis
- summarization
- web_search
- coding
- multi_step

Respond ONLY in valid JSON.

FORMAT:

{
  "query_type": "type",
  "agents": ["agent1", "agent2"],
  "tools": ["tool1", "tool2"],
  "needs_memory": true,
  "complexity": "low/medium/high",
  "reasoning": "short explanation"
}
"""


class PlannerAgent:

    def __init__(self,
                 model="qwen2.5-coder:3b"):
        self.llm = LLMProvider()
        

    def create_plan(self, user_query):

        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": user_query
            }
        ]

        reply = self.llm.generate(messages)

        # Remove markdown if model adds it
        reply = reply.replace(
            "```json",
            ""
        ).replace(
            "```",
            ""
        ).strip()

        try:

            parsed = json.loads(reply)

            return parsed

        except Exception as e:

            return {
                "error": str(e),
                "raw_response": reply
            }


if __name__ == "__main__":

    planner = PlannerAgent()
    executor = AgentExecutor()
    while True:

        query = input("\nUser Query: ")

        if query.lower() == "exit":
            break

        plan = planner.create_plan(query)

        print("\n=== EXECUTION PLAN ===")
        print(json.dumps(plan, indent=2))

        results = executor.execute_plan(
            plan,
            query
        )

        print("\n=== EXECUTION RESULTS ===")

        print(
            json.dumps(
                results,
                indent=2
            )
        )

