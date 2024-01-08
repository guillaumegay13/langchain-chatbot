#!/usr/bin/env python
from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatAnthropic, ChatOpenAI
from langchain.schema import BaseOutputParser
from langserve import add_routes
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    MessagesPlaceholder,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.memory import ConversationBufferWindowMemory
import json

app = FastAPI(
  title="LangChain Server",
  version="1.0",
  description="A simple api server using Langchain's Runnable interfaces",
)

jsonModel = ChatOpenAI(
    #models : https://platform.openai.com/docs/models/gpt-3-5
    #model="gpt-4-1106-preview",
    model="gpt-3.5-turbo-1106",
    model_kwargs={
        "response_format": {
            "type": "json_object"
        }
    }
)

def createRoute(systemTemplate, userTemplate, model, path):
    # Create prompt based on templates
    system_message_prompt = SystemMessagePromptTemplate.from_template(systemTemplate)
    user_message_prompt = HumanMessagePromptTemplate.from_template(userTemplate)
    prompt = ChatPromptTemplate.from_messages(
    [system_message_prompt,
     user_message_prompt]
    )

    add_routes(
        app,
        prompt | model | UnescapedJsonOutputParser(),
        path=path
    )

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
        self.chain.invoke({"input" : prompt_kwargs})

class UnescapedJsonOutputParser(BaseOutputParser[dict]):
    """Parse the output of an LLM call to a dictionary."""

    def parse(self, input: str) -> dict:
        """Parse the output of an LLM call."""
        # Convert the input JSON string to a dictionary
        input_dict = json.loads(input)
        return input_dict

# Generate program prompt templates
GP_system_template = """You are a top-tier personal trainer known for crafting unique, result-driven bodyweight programs rooted in science. Your programs are so premium and detailed that people willingly pay high prices for them."""
GP_user_template = """Design a distinguished four-week {type} training schedule for a {sex} at an {level} level.
He works out {frequency} a week with the aim of {goal}. He's {size}cm tall, weights {weight}.
The regimen should introduce variations as weeks progress.
The result should be a top-notch, detailed, week-by-week regimen worthy of a $100 price tag.
You MUST return the program formatted as JSON object, and nothing else.
The program should contains the following fields: week, weekDescription, day, dayDescription, exercises, exerciceDescription, sets, reps."""

# Create route generate_program
createRoute(GP_system_template, GP_user_template, jsonModel, "/generate_program")

# Provide evidences prompt template
PE_system_template = """You are a worldwide known scientist specialized in body science and fitness training.
Your goal is to provide science-based evidences for a customized fitness training program tailored to the individual's goal of {goal}. 
Your recommendations should be backed by scientific research in the field of body science and should consider physiological data."""
PE_human_template="""Please provide detailed and informative science-based evidences to assist in creating a tailored fitness plan for a person is a {sex}, {size}cm tall, weights {weight}.
The evidences should be related to any factor that influence the fitness plan to achieve the goal of {goal}, such as training method, intensity, number of reps, number of sets, type of exercices, techniques and rest.
Return the evidences as a JSON and feel free to quote meta-analysis, studies, articles.
The JSON should be structured as follow: "evidences": [ "topic": "training", "title": "title", "description": "description", "references": "references" ]"""

createRoute(PE_system_template, PE_human_template, jsonModel, "/generate_evidences")

# Notice that we `return_messages=True` to fit into the MessagesPlaceholder
# Notice that `"chat_history"` aligns with the MessagesPlaceholder name.
# memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
# conversation = LLMChain(
#     llm=jsonModel,
#     prompt=text_prompt,
#     verbose=True,
#     memory=memory
# )

# Generate methods prompt template
GM_system_template = """You are a top-tier personal trainer known for crafting unique, result-driven fitness programs rooted in science.
Based on the science-based evidences that the user will pass you, you will generate the best fitness {type} training methods for a {age} years old {gender} person. 
Explain briefly how those methods are related with the provided scientific evidences and why would they fit perfectly to this person.
Return the training method as a JSON that follows this structure: "methods": [ "name": "name", "description": "description", "execution": "execution", "reference_to_evidence": "reference_to_evidence", "tailored": "tailored" ]"""
GM_human_template="""{evidences}"""

createRoute(GM_system_template, GM_human_template, jsonModel, "/generate_methods")

# add_routes(
#     app,
#     ChatAnthropic(),
#     path="/anthropic",
# )

# model = ChatAnthropic()
# prompt = ChatPromptTemplate.from_template("tell me a joke about {topic}")
# add_routes(
#     app,
#     prompt | model,
#     path="/joke",
# )

### Generate Strenghts and Weeknesses prompt template
SAW_system_template = """You are a top-tier personal trainer known for crafting unique, result-driven fitness programs rooted in science. You are excellent at analysing the strength and weaknesses of your clients."""
SAW_human_template="""Your client is a {age} years old {gender} person, {size}cm tall, weights {weight}kg.
Determine the strengths and weaknesses of this person for a {type} training program. 
Return a JSON that follows this structure: "strengths": [ "name": "name", "description": "description" ], "weaknesses": [ "name": "name", "description": "description"]"""

createRoute(SAW_system_template, SAW_human_template, jsonModel, "/generate_strengths_and_weaknesses")

## Generate weekly program prompt template
WP_system_template = """You are a top-tier personal trainer known for crafting unique, result-driven bodyweight programs rooted in science. Your programs are so premium and detailed that people willingly pay high prices for them.
Based on the training methods the user will send, design a distinguished four-week {type} training schedule for a {gender} at an {level} level.
The schedule should contain exactly {frequency} sessions of training per week and be perfectly adapted to the {level} level.
The regimen should introduce variations as weeks progress.
The result should be a top-notch, detailed, week-by-week regimen worthy of a $100 price tag.
You MUST return the program formatted as JSON object, and nothing else.
The program should contains the following fields: week, weekDescription, [ session, sessionDescription, exercises [ exerciceDescription, sets, reps, reference_to_method ] ]."""
WP_user_template = """{methods}"""

createRoute(WP_system_template, WP_user_template, jsonModel, "/generate_weekly_program")

## Review the program with strenghts and weaknesses
RWSAW_system_template = """You are a top-tier personal trainer known for crafting unique, result-driven training programs that are safe and tailor-made.
The user will provide you a fitness program and the strengths and weaknesses of your client. You need to make sure that the programs is safe and adapted to the client's strengths and weaknesses. 
Return the adapted program as JSON with the following structure: week, weekDescription, [ day, dayDescription, exercises [ exerciceDescription, sets, reps, reference_to_method, adaptability ] ]."""
RWSAW_user_template = """program : {program}, strengths : {strengths}, weaknesses : {weaknesses}"""

createRoute(RWSAW_system_template, RWSAW_user_template, jsonModel, "/review_program")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)