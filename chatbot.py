from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

def chatting(query):
	model = init_chat_model(
		model = "gpt-4.1-mini",
		temperature = 0.2
	)
	response = model.invoke(query)
	print(response.content)

while True:
	chat = input("How may I help you....\n")
	if chat == "exit":
		break
	chatting(chat)

