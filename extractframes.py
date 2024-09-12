import os
from typing import List

import ffmpeg

import utils


input = "/Users/teriyake/Documents/Projects/Coding Projects/Python Projects/Super-Resolution/test media/output/"
output = "/Users/teriyake/Documents/Projects/Coding Projects/Python Projects/Super-Resolution/test media/output/"

framerate = 1
device = utils.get_device()

# List all videos and images as txt in random order
# breaking up videos?
def checkVideoValidity(filePath: str) -> bool:
    if not os.path.exists(filePath):
        return False
    root, extension = os.path.splitext(filePath)
    return extension == ".mp4"

def getVideos(path: str) -> List[str]:
    fileList = []
    for root, dirs, files in os.walk(path):
        dirs = [os.path.join(root, d) for d in dirs]
        files = [os.path.join(root, f) for f in files if checkVideoValidity(os.path.join(root, f))]
        fileList.extend(files)
    return fileList

def convertVideo(path: str) -> int:
    # ffmpeg -hwaccel videotoolbox -i "Valorant 2022.06.02 - 14.30.39.03.DVR.mp4" -vf fps=1 frame_%04d.png
    frames = int(ffmpeg.probe(video)["streams"][0]["nb_frames"])
    hwaccel = {}
    if device == "cuda":
        hwaccel = {"hwaccel": "cuvid"}
    elif device == "mps":
        hwaccel = {"hwaccel": "videotoolbox"}

videoList = []
for path in pathList:
    if not os.path.exists(path):
        continue
    videoList = getVideos(path)
print("Finished reading")

totalFrames = 0
for video in videoList:
    totalFrames += int(ffmpeg.probe(video)["streams"][0]["nb_frames"])

digits = len(str(totalFrames))
framesCompleted = 0
for video in videoList:
    print(f"Converting {video}")
    convertVideo(video, framerate, digits, )