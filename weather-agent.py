import requests
from dotenv import load_dotenv
from dataclasses import dataclass
from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

@dataclass
class Context:
	user_id: str

@dataclass
class ResponseFormat:
	summary: str
	temperature_celcius: float
	humidity: float

@tool('locate_user', description="Locate a user's city based on the context", return_direct=False)
def locate_user(runtime: ToolRuntime[Context]):
	match runtime.context.user_id:
		case 'abc':
			return 'London'
		case 'def':
			return 'Manchester'
		case 'ghi':
			return 'Liverpool'
		case 'jkl':
			return 'Barcelona'
		case _:
			return 'Unknown'

@tool("get_weather", description='Return weather information for a given city', return_direct=False)
def get_weather(city: str):
	response = requests.get(f"https://wttr.in/{city}?format=j1")
	return response.json()

model = init_chat_model('gpt-4.1-mini', temperature=0.3)
checkpointer = InMemorySaver()

agent = create_agent(
	model = model,
	tools = [get_weather, locate_user],
	system_prompt = "You are a helpful weather assistant",
	context_schema = Context,
	response_format = ResponseFormat,
	checkpointer = checkpointer
)

config = {'configurable': {'thread_id': 1}}

response = agent.invoke({
	"messages": [
		{
			"role": "user", "content": "What is the weather like?"
		}
	]},
	config = config,
	context = Context(user_id='def')
)

print(response['structured_response'])
