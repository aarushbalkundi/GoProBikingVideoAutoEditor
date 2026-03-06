import cv2
import numpy as np
# test comment
def ComputeMotionScores(video_path):

    cap = cv2.VideoCapture(video_path)

    ret, prev_frame = cap.read()
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    scores = []
    frame_index = 0

    while True:

        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(
            prev_gray,
            gray,
            None,
            0.5,
            3,
            15,
            3,
            5,
            1.2,
            0
        )

        magnitude = np.mean(np.sqrt(flow[...,0]**2 + flow[...,1]**2))

        scores.append((frame_index, magnitude))

        prev_gray = gray
        frame_index += 1

    cap.release()

    return scores


if __name__ == "__main__":

    video = "../Videos/PCMC2.mp4"

    scores = ComputeMotionScores(video)

    for frame, score in scores[:20]:
        print(frame, score)