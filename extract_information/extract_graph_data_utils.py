import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Type

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field, create_model

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.config import Config


def _field_annotation(type_name: str) -> Any:
    mapping = {
        "str": Optional[str],
        "list[str]": Optional[List[str]],
        "int": Optional[int],
        "float": Optional[float],
        "bool": Optional[bool],
    }
    if type_name not in mapping:
        raise RuntimeError(f"Unsupported schema field type: {type_name}")
    return mapping[type_name]


def _build_attr_model(name: str, fields: Dict[str, str]) -> Type[BaseModel]:
    model_fields: Dict[str, Tuple[Any, Any]] = {}
    for field_name, type_name in fields.items():
        model_fields[field_name] = (
            _field_annotation(type_name),
            Field(default=None, description="optional"),
        )
    return create_model(name, **model_fields)


def _literal_from_values(values: List[str]):
    if not values:
        raise RuntimeError("Schema type list must not be empty.")
    return Literal.__getitem__(tuple(values))


def build_knowledge_graph_model(schema: Dict[str, Any]) -> Type[BaseModel]:
    herb_attr_model = _build_attr_model("HerbAttributes", schema.get("herb_attributes") or {})
    formula_attr_model = _build_attr_model(
        "FormulaAttributes", schema.get("formula_attributes") or {}
    )
    entity_type = _literal_from_values(schema["entity_types"])
    relation_type = _literal_from_values(schema["relation_types"])

    entity_model = create_model(
        "Entity",
        name=(str, Field(description="entity name")),
        type=(entity_type, Field(description="entity type")),
        herb_attributes=(Optional[herb_attr_model], Field(default=None, description="optional")),
        formula_attributes=(
            Optional[formula_attr_model],
            Field(default=None, description="optional"),
        ),
    )
    relation_model = create_model(
        "Relation",
        source=(str, Field(description="source entity name")),
        target=(str, Field(description="target entity name")),
        type=(relation_type, Field(description="relation type")),
    )
    return create_model(
        "KnowledgeGraph",
        entities=(List[entity_model], Field(default_factory=list)),
        relations=(List[relation_model], Field(default_factory=list)),
    )


def _load_schema(conf: Config) -> Dict[str, Any]:
    raw = conf.require("KG_EXTRACTION_SCHEMA")
    schema = json.loads(raw)
    for key in ("entity_types", "relation_types"):
        if key not in schema or not isinstance(schema[key], list):
            raise RuntimeError(f"KG_EXTRACTION_SCHEMA must include a list field '{key}'.")
    return schema


def _load_prompt_template(parser: JsonOutputParser, conf: Config) -> PromptTemplate:
    template = conf.require("KG_EXTRACTION_PROMPT")
    if "{format_instructions}" not in template or "{text}" not in template:
        raise RuntimeError(
            "KG_EXTRACTION_PROMPT must contain {format_instructions} and {text} placeholders."
        )
    return PromptTemplate(
        template=template,
        input_variables=["text"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )


_conf = Config()
_schema = _load_schema(_conf)
KnowledgeGraph = build_knowledge_graph_model(_schema)
parser = JsonOutputParser(pydantic_object=KnowledgeGraph)
prompt = _load_prompt_template(parser, _conf)


if __name__ == "__main__":
    from common.llm import my_llm
    from common.path_utils import get_file_path

    sample_path = get_file_path(
        "clawler/herbal_formulas/Ephedra Decoction (Ma Huang Tang).txt"
    )
    text = Path(sample_path).read_text(encoding="utf-8")
    excerpt = "\n".join(text.splitlines()[:20])

    chain = prompt | my_llm | parser
    result = chain.invoke({"text": excerpt})
    print(json.dumps(result, ensure_ascii=False, indent=2))
