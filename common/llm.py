import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from common.config import Config

conf = Config()

my_llm = ChatOpenAI(
    api_key=conf.MODEL_API_KEY,
    base_url=conf.MODEL_BASE_URL,
    model=conf.MODEL_NAME
)

if __name__ == '__main__':
    messages = [
        HumanMessage(content="introduce yourself!")
    ]
    response = my_llm.invoke(messages)
    print(response.content)