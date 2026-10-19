import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from langchain_core.messages import HumanMessage

from common.llm import my_llm
from langgraph_more_nodes.agent_state import AgentState

llm = my_llm


def llm_direct_out_node(state: AgentState) -> AgentState:
    """Answer the user input directly with the LLM; write it to direct_out and output."""
    messages = [HumanMessage(content=state["input"])]

    answer = llm.invoke(messages).content.strip()

    state["direct_out"] = answer
    state["output"] = answer
    return state


if __name__ == "__main__":
    for text in [
        "How is the weather today?",
        "Write a Python quicksort for me",
        "Introduce yourself in one sentence.",
    ]:
        result = llm_direct_out_node({"input": text})
        print(text)
        print(f"  direct_out: {result['direct_out']}")
        print(f"  output == direct_out: {result['output'] == result['direct_out']}")
        print()
