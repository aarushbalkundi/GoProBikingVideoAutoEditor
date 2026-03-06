import cv2
import numpy as np

prev = cv2.imread("frame_0001.jpg",0)
curr = cv2.imread("frame_0002.jpg",0)

flow = cv2.calcOpticalFlowFarneback(prev,curr,None,0.5,3,15,3,5,1.2,0)

magnitude = np.mean(np.sqrt(flow[...,0]**2 + flow[...,1]**2))

print(magnitude)