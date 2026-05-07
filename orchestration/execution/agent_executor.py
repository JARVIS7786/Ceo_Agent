from agents.retrieval.retrieval_agent import RetrievalAgent
from agents.critic.critic_agent import CriticAgent
from orchestration.state.execution_context import ExecutionContext


# from agents.sql.sql_agent import SQLAgent

class AgentExecutor:

    def __init__(self):
        self.critic_agent = CriticAgent()
        self.retrieval_agent = RetrievalAgent()
        self.available_agents = {
            "retrieval_agent": self.run_retrieval_agent,
            "sql_agent": self.run_sql_agent,
            "critic_agent": self.run_critic_agent,
            "synthesis_agent": self.run_synthesis_agent
        }

    def execute_plan(self, plan, user_query):

        context = ExecutionContext(
            user_query
        )
        execution_results = []


        print("\n=== EXECUTING PLAN ===")

        for agent_name in plan["agents"]:

            print(f"\nRunning: {agent_name}")

            if agent_name in self.available_agents:

                result = self.available_agents[
                    agent_name
                ](context)

                execution_results.append({
                    "agent": agent_name,
                    "result": result
                })

            else:

                execution_results.append({
                    "agent": agent_name,
                    "result": "Agent not implemented"
                })

        return execution_results

    # Placeholder agents

    def run_retrieval_agent(self, context):

        return self.retrieval_agent.retrieve_project_context(
            context.user_query
        )
        
        context.add_agent_output(
            "retrieval_agent",
            result
        )
        return result


    def run_sql_agent(self, query):

        return f"Executed SQL workflow for: {query}"

    def run_critic_agent(self, context):

        return self.context.get_agent_output(
            "retrieval_agent"
        )
        
        result = self.critic_agent.critique(
            context.user_query,
            retieval_output
        )
        context.add_agent_output(
            "critic_agent",
            result
        )
        return result

    def run_synthesis_agent(self, query):

        return f"Synthesized final response for: {query}"
