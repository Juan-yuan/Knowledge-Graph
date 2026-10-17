from langgraph.graph.state import CompiledStateGraph


def output_pic_graph(app: CompiledStateGraph, filename: str = "graph.jpg"):
    try:
        mermaid_code = app.get_graph().draw_mermaid_png()
        with open(filename, 'wb') as f:
            f.write(mermaid_code)
    except Exception as e:
        print(e)