import numpy as np
from MotionScore import ComputeMotionScores

VIDEO_PATH = "/home/wuzya/Projects/GoProAutoEditor/Videos/PCMC2.MP4"
print()
print("got path")
scores = ComputeMotionScores(VIDEO_PATH)
print("got scores")
motion_values = [s[1] for s in scores]
print("got motion values")
threshold = np.percentile(motion_values, 90)
print("computed threshold:", threshold)
highlights = []
print("starting highlight detection")
start = None

for frame, score in scores:

    if score > threshold and start is None:
        start = frame

    elif score <= threshold and start is not None:
        end = frame
        highlights.append((start, end))
        start = None

print("Highlight segments:")
for h in highlights:
    print(h)

