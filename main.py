# main.py

import gradio as gr
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from segmentation import SegmentFloorNode
from depth import DepthEstimationNode
from overlay import OverlayRugNode


# 1. Define LangGraph State
class State(TypedDict, total=False):
    room_image: Annotated[any, add_messages]
    rug_image: Annotated[any, add_messages]
    floor_mask: Annotated[any, add_messages]
    depth_map: Annotated[any, add_messages]
    final_image: Annotated[any, add_messages]


# 2. Construct LangGraph pipeline
graph = StateGraph(State)

graph.add_node("segment", SegmentFloorNode())
graph.add_node("depth", DepthEstimationNode())
graph.add_node("overlay", OverlayRugNode())

# Connect nodes explicitly
graph.add_edge(START, "segment")
graph.add_edge("segment", "depth")
graph.add_edge("depth", "overlay")
graph.add_edge("overlay", END)

# 3. Compile the graph into an app
app = graph.compile()


# 4. Gradio interface
def apply_rug(room_image, rug_image):
    inputs = {
        "room_image": room_image,
        "rug_image": rug_image
    }
    result = app.invoke(inputs)
    return result["final_image"]


iface = gr.Interface(
    fn=apply_rug,
    inputs=[
        gr.Image(type="pil", label="Room Image"),
        gr.Image(type="pil", label="Rug Image")
    ],
    outputs=gr.Image(type="pil", label="Result (Room + Rug)"),
    title="Rug/Carpet Visualizer",
    description="Upload a room photo and a rug image; the rug will be placed on the floor realistically."
)

iface.launch()
