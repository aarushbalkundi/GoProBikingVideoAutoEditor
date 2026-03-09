import numpy as np
from MotionScore import ComputeMotionScores
import matplotlib.pyplot as plt 


VIDEO_PATH = "Videos/PCMC2.MP4"

scores = ComputeMotionScores(VIDEO_PATH)

motion_values = [s[1] for s in scores]

threshold = np.percentile(motion_values, 90)

segments = []
start = None

for frame, score in scores:

    if score > threshold and start is None:
        start = frame

    elif score <= threshold and start is not None:
        end = frame
        segments.append((start, end))
        start = None


# merge nearby segments
merged = []
min_gap = 30   # frames

for seg in segments:

    if not merged:
        merged.append(seg)
        continue

    prev_start, prev_end = merged[-1]
    curr_start, curr_end = seg

    if curr_start - prev_end < min_gap:
        merged[-1] = (prev_start, curr_end)
    else:
        merged.append(seg)


# remove very short segments
final_segments = []

min_length = 50  # frames

for start, end in merged:
    if end - start > min_length:
        final_segments.append((start, end))


print("Final highlight segments:\n")

for seg in final_segments:
    print(seg)

frames = [s[0] for s in scores]
motion = [s[1] for s in scores]

plt.figure(figsize=(12,5))

plt.plot(frames, motion)

plt.title("Motion Score Over Time")
plt.xlabel("Frame Number")
plt.ylabel("Motion Score")

plt.grid(True)

plt.savefig("motion_graph.png")
print("Graph saved as motion_graph.png")