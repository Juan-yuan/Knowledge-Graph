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

        self.ENTITY_INDEX_PATH = get_file_path("langgraph_more_nodes/nero4j_embedding_faiss.index")
        self.ENTITY_ID2TEXT_PATH = get_file_path("langgraph_more_nodes/nero4j_embedding_faiss_id2text.pkl")

    def require(self, key: str) -> str:
        value = getattr(self, key, None) or os.getenv(key)
        if not value:
            raise RuntimeError(f"{key} is missing. Set it in common/.env (see .env.example).")
        return value


if __name__ == "__main__":
    conf = Config()
    print(conf.MODEL_BASE_URL)
