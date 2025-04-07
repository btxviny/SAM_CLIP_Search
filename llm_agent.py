import os
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_openai import ChatOpenAI
import dotenv
dotenv.load_dotenv()

llm = ChatOpenAI(
        api_key = os.environ['OPENAI_API_KEY'],
        model ='gpt-4o-mini',
        temperature = 0.1
)

prompt = PromptTemplate(
    template=""" You are a helpful assistant that reviews user commands and extracts only the object description contained in the command.
    For example the user might say 'Grab the red ball'. Your response should be : 'red ball'.
    User Command: {command}""",
    input_variables=["command"]
)

agent = prompt | llm

def get_object_description(command) -> str:
    return agent.invoke({"command": command}).content