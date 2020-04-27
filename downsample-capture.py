import os

BASE_FOLDER = "T:/dekra/video-classifer/minicap positivos originales/futbol/2020-04-12-12-59-43 - SET 19 - dr2"

# Dataset preparation
downsampling_rate = 2
i = 1
for root, dirs, files in os.walk(BASE_FOLDER):
    for name in files:
        if i < downsampling_rate:
            os.remove(root + os.sep + name)
            i = i+1
        else:
            i = 1

