import os
import subprocess
from typing import List

import utils

pathList = ["D:/Videos/Valorant/"]
output = "D:/Videos/out.mp4"

framerate = 1
device = utils.getDevice()
if device == "cuda":
    device = "cpu"


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

videoList = []
for path in pathList:
    if not os.path.exists(path):
        continue
    videoList = getVideos(path)
for video in videoList.copy(): # creates a copy to avoid infinite loop
    videoList.pop(0) # removes the original
    videoList.append("-i")
    videoList.append(video)
print("Finished reading")

command = ["ffmpeg"]

if device == "cuda": # for hwaccel
    command.extend(["-hwaccel", "cuda"])
elif device == "mps":
    command.extend(["-hwaccel", "videotoolbox"])

command.extend(videoList)
concat = ""
for i in range(len(videoList) // 2): # skip the "-i"
    concat += f"[{i}:v]"
command.extend([
    "-filter_complex", f"{concat}concat=n={len(videoList) // 2}:v=1[v]",
    "-map", "[v]",
    "-an",
    "-tag:v", "hvc1", # for hevc
    "-r", f"{framerate}"
    # "-fps_mode", f"fps={framerate}"
])

if device == "cuda": # for hwaccel
    command.extend([
        "-c:v", "hevc_nvenc",
        "-preset", "slow"
    ])
elif device == "mps":
    command.extend(["-c:v", "hevc_videotoolbox"])
else:
    command.extend(["-c:v", "libx265"])

command.extend(["-q:v", "80"])
    
command.append(output)
print(command)
print("Starting...")
subprocess.call(command)
print("Done!")