import os

from orchestration.llm_provider import LLMProvider


class RetrievalAgent:

    def __init__(self):

        self.llm = LLMProvider()

    def retrieve_project_context(self, query):

        current_dir = os.getcwd()

        collected_data = []

        # Walk project files
        for root, dirs, files in os.walk(current_dir):

            for file in files:

                if file.endswith(
                    (
                        ".py",
                        ".md",
                        ".txt",
                        ".json"
                    )
                ):

                    path = os.path.join(root, file)

                    try:

                        with open(
                            path,
                            "r",
                            encoding="utf-8"
                        ) as f:

                            content = f.read()[:3000]

                        collected_data.append(
                            f"""
FILE: {path}

CONTENT:
{content}
"""
                        )

                    except:
                        pass

        combined_context = "\n".join(
            collected_data[:10]
        )

        messages = [
            {
                "role": "system",
                "content": """
You are a Retrieval Agent.

Your task:
- inspect project files
- identify architecture
- extract important system details
- summarize technical structure
"""
            },
            {
                "role": "user",
                "content": f"""
QUERY:
{query}

PROJECT CONTEXT:
{combined_context}
"""
            }
        ]

        summary = self.llm.generate(messages)

        return summary
