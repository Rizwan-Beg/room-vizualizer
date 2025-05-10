# segmentation.py

from transformers import pipeline
import numpy as np

class SegmentFloorNode:
    def __init__(self):
        # Load a semantic segmentation model (ADE20k) via HF pipeline
        self.segmenter = pipeline(
            "image-segmentation",
            model="nvidia/segformer-b0-finetuned-ade-512-512",
            device=0  # use GPU if available
        )

    def __call__(self, state):
        """
        state: {'room_image': PIL.Image}
        returns: {'floor_mask': numpy.ndarray (2D boolean array)}
        """
        room_img = state["room_image"]
        results = self.segmenter(room_img)  # list of {label, mask}

        floor_mask = None
        for res in results:
            if res["label"].lower() == "floor":
                mask_img = res["mask"].convert("L")
                mask_arr = np.array(mask_img) > 0
                floor_mask = mask_arr
                break

        if floor_mask is None:
            floor_mask = np.zeros((room_img.height, room_img.width), dtype=bool)

        return {"floor_mask": floor_mask}
