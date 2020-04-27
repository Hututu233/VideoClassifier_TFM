import cv2
import os
print(cv2.__version__)

BASE_FOLDER = "T:/dekra/video-classifer"
VIDEO_NAME = "tour-france-360p.mp4"
downsampling_rate = 4
i = 1

vidcap = cv2.VideoCapture(BASE_FOLDER+os.sep+'videos'+os.sep+VIDEO_NAME)
success, image = vidcap.read()
count = 0
while success:
  if i < downsampling_rate:
    i = i + 1
  else:
    cv2.imwrite(BASE_FOLDER+os.sep+'frames'+os.sep+"set 3 frame%d.jpg" % count, image)  # save frame as JPEG file
    i = 1
  success, image = vidcap.read()
  print('Read a new frame: ', success)
  count += 1
