import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.path_utils import get_file_path


def _load_env() -> None:
    # Reload on every Config() so Streamlit / uvicorn pick up .env edits without a full process restart
    load_dotenv(get_file_path(".env"), override=True)
    load_dotenv(get_file_path("common/.env"), override=True)


_load_env()


class Config:
    def __init__(self):
        _load_env()
        self.MODEL_API_KEY = os.getenv("MODEL_API_KEY")
        self.MODEL_BASE_URL = os.getenv("MODEL_BASE_URL")
        self.MODEL_NAME = os.getenv("MODEL_NAME")

        self.NEO4J_URI = os.getenv("NEO4J_URI")
        self.NEO4J_USER = os.getenv("NEO4J_USER")
        self.NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
        self.NEO4J_CYPHER_PROGRESS_DESC = os.getenv("NEO4J_CYPHER_PROGRESS_DESC")
        self.NEO4J_RUN_CYPHER_DOC = os.getenv("NEO4J_RUN_CYPHER_DOC")
        self.NEO4J_RUN_MULTIPLE_CYPHER_DOC = os.getenv("NEO4J_RUN_MULTIPLE_CYPHER_DOC")
        self.NEO4J_IMPORT_BATCH_SIZE = os.getenv("NEO4J_IMPORT_BATCH_SIZE", "500")

        self.JIMENG_AK = os.getenv("JIMENG_AK")
        self.JIMENG_SK = os.getenv("JIMENG_SK")

        self.EMBEDDING_MODEL_PATH = os.getenv("EMBEDDING_MODEL_PATH")

        self.HERBS_BASE_URL = os.getenv("HERBS_BASE_URL")
        self.HERBS_LIST_URL = os.getenv("HERBS_LIST_URL")
        self.FORMULAS_LIST_URL = os.getenv("FORMULAS_LIST_URL")
        self.CRAWLER_USER_AGENT = os.getenv("CRAWLER_USER_AGENT")
        self.CRAWLER_CONTENT_SELECTOR = os.getenv("CRAWLER_CONTENT_SELECTOR")
        self.CRAWLER_LIST_LINK_SELECTOR = os.getenv("CRAWLER_LIST_LINK_SELECTOR")
        self.CRAWLER_DETAIL_REMOVE_SELECTORS = os.getenv("CRAWLER_DETAIL_REMOVE_SELECTORS")
        self.CRAWLER_DETAIL_UNWRAP_SELECTORS = os.getenv("CRAWLER_DETAIL_UNWRAP_SELECTORS")
        self.CRAWLER_DETAIL_TITLE_SELECTOR = os.getenv("CRAWLER_DETAIL_TITLE_SELECTOR")
        self.CRAWLER_BATCH_SIZE = os.getenv("CRAWLER_BATCH_SIZE", "50")
        self.CRAWLER_MAX_RETRIES = os.getenv("CRAWLER_MAX_RETRIES", "3")
        self.CRAWLER_TRANSLATE_WORKERS = os.getenv("CRAWLER_TRANSLATE_WORKERS", "10")
        self.CRAWLER_DETAIL_WORKERS = os.getenv("CRAWLER_DETAIL_WORKERS", "10")
        self.CRAWLER_REQUEST_DELAY_MIN = os.getenv("CRAWLER_REQUEST_DELAY_MIN", "0.5")
        self.CRAWLER_REQUEST_DELAY_MAX = os.getenv("CRAWLER_REQUEST_DELAY_MAX", "1.5")
        self.CRAWLER_CHUNK_SIZE = os.getenv("CRAWLER_CHUNK_SIZE", "4000")

        # Private prompts / schema (common/.env only; not committed)
        self.CRAWLER_HERBS_LIST_PROMPT = os.getenv("CRAWLER_HERBS_LIST_PROMPT")
        self.CRAWLER_FORMULAS_LIST_PROMPT = os.getenv("CRAWLER_FORMULAS_LIST_PROMPT")
        self.CRAWLER_HERBS_DETAIL_PROMPT = os.getenv("CRAWLER_HERBS_DETAIL_PROMPT")
        self.CRAWLER_FORMULAS_DETAIL_PROMPT = os.getenv("CRAWLER_FORMULAS_DETAIL_PROMPT")
        self.KG_EXTRACTION_PROMPT = os.getenv("KG_EXTRACTION_PROMPT")
        self.KG_EXTRACTION_PROJECT_INFO = os.getenv("KG_EXTRACTION_PROJECT_INFO")
        self.KG_EXTRACTION_SCHEMA = os.getenv("KG_EXTRACTION_SCHEMA")
        self.KG_FINETUNE_INSTRUCTION = os.getenv("KG_FINETUNE_INSTRUCTION")
        self.KG_TRANSLATE_PROMPT = os.getenv("KG_TRANSLATE_PROMPT")
        self.KG_EXTRACT_WORKERS = os.getenv("KG_EXTRACT_WORKERS", "5")
        self.KG_EXTRACT_MAX_RETRIES = os.getenv("KG_EXTRACT_MAX_RETRIES", "3")
        self.ZHONGYI_INTENT_PROMPT = os.getenv("ZHONGYI_INTENT_PROMPT")
        self.ENTITY_EXTRACTION_PROMPT = os.getenv("ENTITY_EXTRACTION_PROMPT")
        self.CYPHER_GENERATION_PROMPT = os.getenv("CYPHER_GENERATION_PROMPT")
        self.NEO4J_ANSWER_PROMPT = os.getenv("NEO4J_ANSWER_PROMPT")
        self.MAX_CYPHER_CHECKS = os.getenv("MAX_CYPHER_CHECKS")
        # Number of past messages fed to the semantic transcription node
        self.HISTORY_NUM = int(os.getenv("HISTORY_NUM", "6"))
        self.LLM_DIRECT_OUT_SYSTEM_PROMPT = os.getenv("LLM_DIRECT_OUT_SYSTEM_PROMPT", "")

        # Private entity / Cypher tech points
        self.TCM_METADATA = os.getenv("TCM_METADATA")
        self.ENTITY_LABEL_CONFIG = os.getenv("ENTITY_LABEL_CONFIG")
        self.ENTITY_INPUT_TO_MATCHED_KEY = os.getenv("ENTITY_INPUT_TO_MATCHED_KEY")
        self.ENTITY_MATCHED_KEY_TO_LABEL = os.getenv("ENTITY_MATCHED_KEY_TO_LABEL")
        self.ENTITY_MATCH_TOP_K = os.getenv("ENTITY_MATCH_TOP_K", "5")
        self.ENTITY_MATCH_THRESHOLD = os.getenv("ENTITY_MATCH_THRESHOLD", "0.6")
        self.ENTITY_NEAR_DUPLICATE_THRESHOLD = os.getenv("ENTITY_NEAR_DUPLICATE_THRESHOLD", "0.15")
        self.CYPHER_WRITE_KEYWORDS = os.getenv("CYPHER_WRITE_KEYWORDS")

        # Private TCM Neo4j import settings
        self.TCM_NEO4J_PROJECT = os.getenv("TCM_NEO4J_PROJECT")
        self.TCM_IMPORT_ENTITY_LABELS = os.getenv("TCM_IMPORT_ENTITY_LABELS")
        self.TCM_IMPORT_ATTRIBUTE_KEYS = os.getenv("TCM_IMPORT_ATTRIBUTE_KEYS")
        self.TCM_IMPORT_RELATION_SCHEMA = os.getenv("TCM_IMPORT_RELATION_SCHEMA")
        self.TCM_IMPORT_PROJECT_INFO = os.getenv("TCM_IMPORT_PROJECT_INFO")

        # FastAPI server
        self.FASTAPI_HOST = os.getenv("FASTAPI_HOST", "0.0.0.0")
        self.FASTAPI_PORT = int(os.getenv("FASTAPI_PORT", "8000"))
        self.FASTAPI_TITLE = os.getenv("FASTAPI_TITLE", "TCM Knowledge Graph System")
        self.FASTAPI_DESCRIPTION = os.getenv(
            "FASTAPI_DESCRIPTION",
            "TCM knowledge graph based on FastAPI and LangGraph",
        )
        self.FASTAPI_ERROR_OUTPUT = os.getenv("FASTAPI_ERROR_OUTPUT", "System error, please try again!")
        self.FASTAPI_HEALTH_MESSAGE = os.getenv(
            "FASTAPI_HEALTH_MESSAGE",
            "TCM Knowledge Graph System is running normally",
        )
        self.FASTAPI_KEY_STATE_FIELDS = os.getenv("FASTAPI_KEY_STATE_FIELDS", "[]")

        # FastAPI client / tests
        self.API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
        self.API_TEST_USER_ID = os.getenv("API_TEST_USER_ID", "user-123")
        self.API_HEALTH_TIMEOUT = int(os.getenv("API_HEALTH_TIMEOUT", "10"))
        self.API_PROCESS_TIMEOUT = int(os.getenv("API_PROCESS_TIMEOUT", "120"))
        self.API_MISSING_FIELD_TIMEOUT = int(os.getenv("API_MISSING_FIELD_TIMEOUT", "10"))

        # Streamlit UI copy (private)
        self.STREAMLIT_PORT = int(os.getenv("STREAMLIT_PORT", "8501"))
        self.STREAMLIT_TAGLINE = os.getenv("STREAMLIT_TAGLINE", "")
        self.STREAMLIT_EXAMPLE_QUESTIONS = os.getenv("STREAMLIT_EXAMPLE_QUESTIONS", "[]")
        self.STREAMLIT_SIDEBAR_HOWTO = os.getenv("STREAMLIT_SIDEBAR_HOWTO", "[]")
        self.STREAMLIT_DISCLAIMER = os.getenv("STREAMLIT_DISCLAIMER", "")

        self.ENTITY_INDEX_PATH = get_file_path("langgraph_nodes/nero4j_embedding_faiss.index")
        self.ENTITY_ID2TEXT_PATH = get_file_path("langgraph_nodes/nero4j_embedding_faiss_id2text.pkl")

    def require(self, key: str) -> str:
        value = getattr(self, key, None) or os.getenv(key)
        if not value:
            raise RuntimeError(f"{key} is missing. Set it in common/.env (see .env.example).")
        return value

    def require_json(self, key: str):
        return json.loads(self.require(key))

    def get_json(self, key: str, default=None):
        raw = getattr(self, key, None) or os.getenv(key)
        if not raw:
            return default
        return json.loads(raw)


if __name__ == "__main__":
    conf = Config()
    print(json.loads(conf.require("TCM_METADATA")).keys())
