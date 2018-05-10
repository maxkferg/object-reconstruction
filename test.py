import os
import cv2
import time
import masking
from masking import draw_mask_on_video





infile = "input.mp4"
outfile = "output.avi"
model = masking.load_model()
print("Loaded model")

cap = cv2.VideoCapture(infile)
w = int(cap.get(3))
h = int(cap.get(4))
#output = cv2.VideoWriter(outfile, cv2.VideoWriter_fourcc('M','J','P','G'), 10, (w,h))
#print("Output writer is: ",output)

print("Drawing mask on video")
draw_mask_on_video(model, cap, outfile)

cap.release()
output.release()
