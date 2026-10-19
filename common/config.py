import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.path_utils import get_file_path

load_dotenv(get_file_path(".env"))
load_dotenv(get_file_path("common/.env"), override=True)


class Config:
    def __init__(self):
        self.MODEL_API_KEY = os.getenv("MODEL_API_KEY")
        self.MODEL_BASE_URL = os.getenv("MODEL_BASE_URL")
        self.MODEL_NAME = os.getenv("MODEL_NAME")

        self.NEO4J_URI = os.getenv("NEO4J_URI")
        self.NEO4J_USER = os.getenv("NEO4J_USER")
        self.NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
        self.NEO4J_CYPHER_PROGRESS_DESC = os.getenv("NEO4J_CYPHER_PROGRESS_DESC")
        self.NEO4J_RUN_CYPHER_DOC = os.getenv("NEO4J_RUN_CYPHER_DOC")
        self.NEO4J_RUN_MULTIPLE_CYPHER_DOC = os.getenv("NEO4J_RUN_MULTIPLE_CYPHER_DOC")

        self.JIMENG_AK = os.getenv("JIMENG_AK")
        self.JIMENG_SK = os.getenv("JIMENG_SK")

        self.EMBEDDING_MODEL_PATH = os.getenv("EMBEDDING_MODEL_PATH")

        self.HERBS_BASE_URL = os.getenv("HERBS_BASE_URL")
        self.HERBS_LIST_URL = os.getenv("HERBS_LIST_URL")
        self.FORMULAS_LIST_URL = os.getenv("FORMULAS_LIST_URL")
        self.CRAWLER_USER_AGENT = os.getenv("CRAWLER_USER_AGENT")

        # Private prompts / schema (common/.env only; not committed)
        self.CRAWLER_HERBS_LIST_PROMPT = os.getenv("CRAWLER_HERBS_LIST_PROMPT")
        self.CRAWLER_FORMULAS_LIST_PROMPT = os.getenv("CRAWLER_FORMULAS_LIST_PROMPT")
        self.CRAWLER_HERBS_DETAIL_PROMPT = os.getenv("CRAWLER_HERBS_DETAIL_PROMPT")
        self.CRAWLER_FORMULAS_DETAIL_PROMPT = os.getenv("CRAWLER_FORMULAS_DETAIL_PROMPT")
        self.KG_EXTRACTION_PROMPT = os.getenv("KG_EXTRACTION_PROMPT")
        self.KG_EXTRACTION_PROJECT_INFO = os.getenv("KG_EXTRACTION_PROJECT_INFO")
        self.KG_EXTRACTION_SCHEMA = os.getenv("KG_EXTRACTION_SCHEMA")
        self.KG_FINETUNE_INSTRUCTION = os.getenv("KG_FINETUNE_INSTRUCTION")
        self.ZHONGYI_INTENT_PROMPT = os.getenv("ZHONGYI_INTENT_PROMPT")
        self.ENTITY_EXTRACTION_PROMPT = os.getenv("ENTITY_EXTRACTION_PROMPT")
        self.CYPHER_GENERATION_PROMPT = os.getenv("CYPHER_GENERATION_PROMPT")
        self.NEO4J_ANSWER_PROMPT = os.getenv("NEO4J_ANSWER_PROMPT")
        self.MAX_CYPHER_CHECKS = os.getenv("MAX_CYPHER_CHECKS")

        # Private TCM Neo4j import settings
        self.TCM_NEO4J_PROJECT = os.getenv("TCM_NEO4J_PROJECT")
        self.TCM_IMPORT_ENTITY_LABELS = os.getenv("TCM_IMPORT_ENTITY_LABELS")
        self.TCM_IMPORT_ATTRIBUTE_KEYS = os.getenv("TCM_IMPORT_ATTRIBUTE_KEYS")
        self.TCM_IMPORT_RELATION_SCHEMA = os.getenv("TCM_IMPORT_RELATION_SCHEMA")
        self.TCM_IMPORT_PROJECT_INFO = os.getenv("TCM_IMPORT_PROJECT_INFO")

        self.ENTITY_INDEX_PATH = get_file_path("langgraph_more_nodes/nero4j_embedding_faiss.index")
        self.ENTITY_ID2TEXT_PATH = get_file_path("langgraph_more_nodes/nero4j_embedding_faiss_id2text.pkl")

        self.TCM_METADATA = open(get_file_path("common/tcm_metadata.json"), "r", encoding="utf-8").read()

    def require(self, key: str) -> str:
        value = getattr(self, key, None) or os.getenv(key)
        if not value:
            raise RuntimeError(f"{key} is missing. Set it in common/.env (see .env.example).")
        return value

    def require_json(self, key: str):
        return json.loads(self.require(key))



if __name__ == "__main__":
    conf = Config()
    print(conf.TCM_METADATA)
