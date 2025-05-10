# depth.py

from transformers import pipeline
import numpy as np
from PIL import Image

class DepthEstimationNode:
    def __init__(self):
        # Load DPT model for depth estimation
        self.depth_pipe = pipeline(
            task="depth-estimation",
            model="Intel/dpt-large",
            device=0  # use GPU if available
        )

    def __call__(self, state):
        """
        state: {'room_image': PIL.Image}
        returns: {'depth_map': numpy.ndarray (H x W float32)}
        """
        room_img = state["room_image"]

        result = self.depth_pipe(room_img)  # run depth estimation

        # 'depth' is a PIL.Image; convert it to np.array
        depth_img = result["depth"]
        depth_arr = np.array(depth_img).astype(np.float32)

        return {"depth_map": depth_arr}
