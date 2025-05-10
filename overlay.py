# overlay.py

import numpy as np
from PIL import Image

class OverlayRugNode:
    def __call__(self, state):
        """
        Inputs in `state`:
            - 'room_image': PIL.Image
            - 'rug_image': PIL.Image
            - 'floor_mask': numpy.ndarray (bool)
            - 'depth_map': numpy.ndarray (float32)
        Returns:
            - {'final_image': PIL.Image}
        """

        room_img = state["room_image"].convert("RGBA")
        rug_img = state["rug_image"].convert("RGBA")
        floor_mask = state["floor_mask"]
        depth_map = state["depth_map"]

        # Resize rug to approximately fit the floor area
        rug_resized = rug_img.resize((room_img.width, room_img.height))

        # Create empty canvas (transparent)
        composite = Image.new("RGBA", room_img.size)

        # Use floor mask to place rug
        rug_array = np.array(rug_resized)
        mask_indices = floor_mask

        # Convert to composite image
        composite_array = np.array(composite)
        room_array = np.array(room_img)

        # Overlay rug only on floor regions
        for c in range(4):  # RGBA channels
            composite_array[..., c] = np.where(
                mask_indices,
                rug_array[..., c],
                room_array[..., c]
            )

        final_image = Image.fromarray(composite_array, mode="RGBA")

        return {"final_image": final_image}
