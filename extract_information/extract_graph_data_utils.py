import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Type

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field, create_model
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.config import Config
from common.llm import my_llm


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


def load_existing_results(save_path: str) -> dict:
    """Load an existing JSON result file for resume support."""
    if os.path.exists(save_path):
        with open(save_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                if isinstance(data, dict) and "results" in data:
                    return data
            except json.JSONDecodeError:
                pass
    return {"results": []}


def save_results(data: Any, save_path: str) -> None:
    """Persist extraction or finetune data to JSON."""
    parent = os.path.dirname(save_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def extract_tcm_knowledge(text: str) -> dict:
    """Extract entities and relations from a single TCM text."""
    chain = prompt | my_llm | parser
    result = chain.invoke({"text": text})
    if hasattr(result, "model_dump"):
        return result.model_dump()
    return result


def extract_from_folder(folder_path: str, save_path: str, finetune_save_path: str) -> dict:
    """
    Extract knowledge from every .txt file in a folder.

    Skips files already present in save_path so runs can be resumed.
    Also appends instruction/input/output records to finetune_save_path.
    """
    all_results = load_existing_results(save_path)
    processed_files = {r["filename"] for r in all_results["results"]}

    finetune_data: List[dict] = []
    if os.path.exists(finetune_save_path):
        with open(finetune_save_path, "r", encoding="utf-8") as f:
            try:
                loaded = json.load(f)
                if isinstance(loaded, list):
                    finetune_data = loaded
            except json.JSONDecodeError:
                pass

    finetune_instruction = _conf.require("KG_FINETUNE_INSTRUCTION")
    txt_files = [f for f in os.listdir(folder_path) if f.endswith(".txt")]
    print(f"Found {len(txt_files)} text file(s).")

    for filename in tqdm(txt_files, desc="Extracting"):
        if filename in processed_files:
            tqdm.write(f"Skipping already processed file: {filename}")
            continue

        file_path = os.path.join(folder_path, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        if not text:
            tqdm.write(f"Empty file skipped: {filename}")
            continue

        try:
            result_dict = extract_tcm_knowledge(text)
            all_results["results"].append(
                {"filename": filename, "extract_dict": result_dict}
            )
            save_results(all_results, save_path)
            tqdm.write(f"Saved result: {filename}")

            finetune_data.append(
                {
                    "instruction": finetune_instruction,
                    "input": text,
                    "output": json.dumps(result_dict, ensure_ascii=False, indent=2),
                }
            )
            save_results(finetune_data, finetune_save_path)
            tqdm.write(f"Appended finetune record: {filename}")
        except Exception as e:
            tqdm.write(f"Failed: {filename}: {e}")

    print(f"Done. Extracted {len(all_results['results'])} file result(s).")
    return all_results


if __name__ == "__main__":
    from common.path_utils import get_file_path

    sample_path = get_file_path(
        "clawler/herbal_formulas/Ephedra Decoction (Ma Huang Tang).txt"
    )
    text = Path(sample_path).read_text(encoding="utf-8")
    excerpt = "\n".join(text.splitlines()[:20])
    result = extract_tcm_knowledge(excerpt)
    print(json.dumps(result, ensure_ascii=False, indent=2))
