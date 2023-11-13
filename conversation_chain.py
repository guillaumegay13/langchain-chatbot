from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import ChatPromptTemplate
from langchain.schema import BaseOutputParser

# Load API key from .env file
import os
from os.path import join, dirname
from dotenv import load_dotenv
dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)
api_key = os.environ.get('API_KEY')

# Initialize the language model
chat = ChatOpenAI(api_key=api_key)

messages = ([
    SystemMessage(
        content="You are a helpful assistant."
    ),
    HumanMessage(
        content="How many people live in France?"
    )
])

print(chat(messages))

prompt = PromptTemplate.from_template("What is a good name for a company that makes {product}?")
prompt.format(product="colorful socks")

template = "You are a helpful assistant that translates {input_language} to {output_language}."
human_template = "{text}"

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", template),
    ("human", human_template),
])

chat_prompt.format_messages(input_language="English", output_language="French", text="I love programming.")