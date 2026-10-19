import json
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field

from common.config import Config
from common.llm import my_llm
from langgraph_more_nodes.agent_state import AgentState

conf = Config()
ENTITY_EXTRACTION_PROMPT = conf.require("ENTITY_EXTRACTION_PROMPT")

# Entity label -> AgentState field
LABEL_TO_STATE_KEY = {
    "Effect": "user_input_effects",
    "Disease": "user_input_diseases",
    "Symptom": "user_input_symptoms",
    "Formula": "user_input_formulas",
    "Herb": "user_input_herbs",
    "Source": "user_input_sources",
}

LABEL_DESCRIPTIONS = {
    label["name"]: label["description"]
    for label in json.loads(conf.TCM_METADATA)["labels"]
    if label["name"] in LABEL_TO_STATE_KEY
}


class UserInputEntities(BaseModel):
    """Entities mentioned in the user input, grouped by type."""

    effects: List[str] = Field(default_factory=list, description=LABEL_DESCRIPTIONS["Effect"])
    diseases: List[str] = Field(default_factory=list, description=LABEL_DESCRIPTIONS["Disease"])
    symptoms: List[str] = Field(default_factory=list, description=LABEL_DESCRIPTIONS["Symptom"])
    formulas: List[str] = Field(default_factory=list, description=LABEL_DESCRIPTIONS["Formula"])
    herbs: List[str] = Field(default_factory=list, description=LABEL_DESCRIPTIONS["Herb"])
    sources: List[str] = Field(default_factory=list, description=LABEL_DESCRIPTIONS["Source"])


# function_calling is the most widely supported mode across OpenAI-compatible providers
llm = my_llm.with_structured_output(UserInputEntities, method="function_calling")


def _dedupe(items: List[str]) -> List[str]:
    seen = set()
    result = []
    for item in items:
        item = item.strip()
        if item and item.lower() not in seen:
            seen.add(item.lower())
            result.append(item)
    return result


def extract_entity_from_user_input_node(state: AgentState) -> AgentState:
    """Extract Symptom/Disease/Formula/Herb/Effect/Source entities from the user input."""
    user_input = state["input"]
    entity_types = "\n".join(f"- {name}: {desc}" for name, desc in LABEL_DESCRIPTIONS.items())
    messages = [
        SystemMessage(content=f"{ENTITY_EXTRACTION_PROMPT}\n\nEntity types:\n{entity_types}"),
        HumanMessage(content=user_input),
    ]

    entities = llm.invoke(messages) or UserInputEntities()

    state["user_input_effects"] = _dedupe(entities.effects)
    state["user_input_diseases"] = _dedupe(entities.diseases)
    state["user_input_symptoms"] = _dedupe(entities.symptoms)
    state["user_input_formulas"] = _dedupe(entities.formulas)
    state["user_input_herbs"] = _dedupe(entities.herbs)
    state["user_input_sources"] = _dedupe(entities.sources)
    return state


if __name__ == "__main__":
    for text in [
        "What are the effects of Astragalus (Huang Qi)?",
        "What Chinese herbs can help with insomnia and vivid dreams?",
        "Which formulas from the Shang Han Lun treat fever and aversion to cold caused by exterior wind-cold?",
        "How is the weather today?",
    ]:
        result = extract_entity_from_user_input_node({"input": text})
        print(text)
        for key in LABEL_TO_STATE_KEY.values():
            if result[key]:
                print(f"  {key}: {result[key]}")
