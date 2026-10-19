from typing import Any, List, TypedDict


class AgentState(TypedDict):
    # Input
    input: str
    # Whether the query is TCM-related
    is_zhongyi_intent: bool
    # Direct reply (skip graph path)
    direct_out: str
    # Entities extracted from user input
    user_input_effects: List[str]
    user_input_diseases: List[str]
    user_input_symptoms: List[str]
    user_input_formulas: List[str]
    user_input_herbs: List[str]
    user_input_sources: List[str]
    # Matched entities
    matched_effects: List[str]
    matched_diseases: List[str]
    matched_symptoms: List[str]
    matched_formulas: List[str]
    matched_herbs: List[str]
    matched_sources: List[str]
    # Cypher queries
    cypher_query: List[str]
    is_all_validate_cypher: bool
    # Number of Cypher checks so far (caps the regenerate loop)
    cypher_check_count: int
    cypher_results: List[Any]
    neo4j_answer: str
    # Output
    output: str
