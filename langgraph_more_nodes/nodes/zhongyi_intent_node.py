import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from langchain_core.messages import SystemMessage, HumanMessage

from common.config import Config
from common.llm import my_llm
from langgraph_more_nodes.agent_state import AgentState

llm = my_llm
conf = Config()
ZHONGYI_INTENT_PROMPT = conf.require("ZHONGYI_INTENT_PROMPT")


def zhongyi_intent_node(state: AgentState) -> AgentState:
    """Decide whether the user input is TCM-related; write the result to is_zhongyi_intent."""
    user_input = state["input"]
    messages = [
        SystemMessage(content=ZHONGYI_INTENT_PROMPT),
        HumanMessage(content=user_input),
    ]

    response = llm.invoke(messages).content.strip()

    state["is_zhongyi_intent"] = response.lower().startswith("yes")
    return state


if __name__ == "__main__":
    for text in [
        "What are the effects of Astragalus (Huang Qi)?",
        "What Chinese herbs can help with insomnia and vivid dreams?",
        "How is the weather today?",
        "Write a Python quicksort for me",
    ]:
        result = zhongyi_intent_node({"input": text})
        print(text, "->", result["is_zhongyi_intent"])
