import cv2
import numpy as np

def ComputeMotionScores(video_path, frame_skip=5):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise Exception("Error: Could not open video file")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print("FPS:", fps)
    print("Total frames:", total_frames)

    ret, prev_frame = cap.read()

    if not ret:
        raise Exception("Error: Could not read first frame")

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    # resize to make processing faster
    prev_gray = cv2.resize(prev_gray, (320, 180))

    # crop center to remove GoPro distortion
    h, w = prev_gray.shape
    prev_gray = prev_gray[int(h*0.2):int(h*0.8), int(w*0.2):int(w*0.8)]

    scores = []
    frame_index = 1

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        if frame_index % frame_skip != 0:
            frame_index += 1
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        gray = cv2.resize(gray, (320, 180))

        h, w = gray.shape
        gray = gray[int(h*0.2):int(h*0.8), int(w*0.2):int(w*0.8)]

        # frame differencing (very fast motion detection)
        diff = cv2.absdiff(prev_gray, gray)

        motion_score = diff.mean()

        scores.append((frame_index, motion_score))

        prev_gray = gray

        if frame_index % 500 == 0:
            print("Processed frame:", frame_index)

        frame_index += 1

    cap.release()

    print("Finished processing")

    return scores


if __name__ == "__main__":

    VIDEO_PATH = "Videos/PCMC2.MP4"

    scores = ComputeMotionScores(VIDEO_PATH)

    print("\nFirst 20 motion scores:\n")

    for frame, score in scores[:20]:
        print(frame, score)