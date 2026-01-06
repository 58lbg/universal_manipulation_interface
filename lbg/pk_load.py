import os
import pickle

base_dir = os.path.dirname(__file__)
tag_detection = os.path.join(base_dir, "../100GOPRO/demos/mapping/tag_detection.pkl")
tag_detection_results = pickle.load(open(tag_detection, 'rb'))
print(tag_detection_results)