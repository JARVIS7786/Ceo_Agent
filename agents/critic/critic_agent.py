from orchestration.llm_provider import LLMProvider


SYSTEM_PROMPT = """
You are a Critic Agent.

Your task:
- analyze outputs critically
- identify weaknesses
- detect hallucinations
- identify architectural flaws
- suggest improvements
- evaluate completeness

Be highly analytical and technical.
"""


class CriticAgent:

    def __init__(self):

        self.llm = LLMProvider()

    def critique(self,
                 user_query,
                 generated_output):

        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": f"""
USER QUERY:
{user_query}

GENERATED OUTPUT:
{generated_output}

Critique this output thoroughly.
"""
            }
        ]

        critique = self.llm.generate(messages)

        return critique
