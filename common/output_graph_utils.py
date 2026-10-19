from pathlib import Path

from langgraph.graph.state import CompiledStateGraph

# Knowledge-Graph/picture (resolved from this file, not the process cwd)
PICTURE_DIR = Path(__file__).resolve().parents[1] / "picture"


def output_pic_graph(app: CompiledStateGraph, filename: str = "graph.jpg") -> str:
    PICTURE_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PICTURE_DIR / Path(filename).name
    mermaid_png = app.get_graph().draw_mermaid_png()
    output_path.write_bytes(mermaid_png)
    print(f"Graph image saved to: {output_path}")
    return str(output_path)
