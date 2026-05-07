import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()


class LLMProvider:

    def __init__(
        self,
        model="llama-3.3-70b-versatile"
    ):

        self.client = Groq(
            api_key=os.getenv("GROQ_API_KEY")
        )

        self.model = model

    def generate(self, messages):

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0
        )

        return response.choices[0].message.content

