from pathlib import Path

import numpy as np
import cv2
from umi.common.cv_util import draw_predefined_mask, draw_predefined_mask1


video_dir = Path(__file__).resolve().parent
mask_write_path = video_dir.joinpath('slam_mask.png')
slam_mask = np.zeros((2028, 2704), dtype=np.uint8)
# slam_mask = draw_predefined_mask(
#     slam_mask, color=255, mirror=True, gripper=False, finger=True)
slam_mask = draw_predefined_mask1(
    slam_mask, color=255, mirror=True, gripper=False, finger=True)
cv2.imwrite(str(mask_write_path.absolute()), slam_mask)