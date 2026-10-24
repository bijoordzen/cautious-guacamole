from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="text-generation",
    max_new_tokens=500,
    provider="auto"
)

model = ChatHuggingFace(llm=llm)

result = model.invoke("what is differention of 2x sqaure")

print(result.content)
