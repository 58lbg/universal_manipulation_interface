import os
import pickle

base_dir = os.path.dirname(__file__)
tag_detection = os.path.join(base_dir, "../100GOPRO/demos/gripper_calibration_C3441350089206_2026.01.06_19.04.07.612367/tag_detection.pkl")
tag_detection_results = pickle.load(open(tag_detection, 'rb'))
print(tag_detection_results)