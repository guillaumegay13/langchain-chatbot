from langchain.prompts import ChatPromptTemplate
from langchain.schema import BaseOutputParser
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
import json

class UnescapedJsonOutputParser(BaseOutputParser[dict]):
    """Parse the output of an LLM call to a dictionary."""

    def parse(self, input: str) -> dict:
        """Parse the output of an LLM call."""
        # Convert the input JSON string to a dictionary
        input_dict = json.loads(input)
        return input_dict

class Chain:
    def __init__(self, systemTemplate, userTemplate, model):
        # Create chain without JSON output
        system_message_prompt = SystemMessagePromptTemplate.from_template(systemTemplate)
        user_message_prompt = HumanMessagePromptTemplate.from_template(userTemplate)
        self.prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt,
        user_message_prompt]
        )
        self.model = model
        self.chain = self.prompt | self.model | UnescapedJsonOutputParser()

    def invoke(self, **prompt_kwargs):
        # Take a dictionary input argument
        return self.chain.invoke(prompt_kwargs)
