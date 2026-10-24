#import
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

#load .env
load_dotenv()

#llm create using gpt-oss-120b
llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="text-generation",
    max_new_tokens=500,
    provider="auto"
)

#model create(chathuggingface model)
model = ChatHuggingFace(llm=llm)

#loop(with multi turn chat simple code without memory)
while True:
    user_input = input("You: ")
    if user_input.strip().lower() in ["exit", "quit"]:
        break

    result = model.invoke(user_input)
    print(f"AI: {result.content}")
