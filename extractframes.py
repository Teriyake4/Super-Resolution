import io
import os
import subprocess
from typing import List
from PIL import Image

import ffmpeg

import utils


# input = "test media/out.mp4"
# output = "test media/output/"
input = "D:/Videos/out.mp4"
output = "D:/Videos/output/"

framerate = 1
device = utils.getDevice()

# List all videos and images as txt in random order
# breaking up videos?
# ffmpeg -hwaccel videotoolbox -i "Valorant 2022.06.02 - 14.30.39.03.DVR.mp4" -vf fps=1 frame_%04d.png
frames = int(ffmpeg.probe(input)["streams"][0]["nb_frames"])
command = ["ffmpeg"]
if device == "cuda": # for hwaccel
    command.extend(["-hwaccel", "cuda"])
elif device == "mps":
    command.extend(["-hwaccel", "videotoolbox"])
command.extend([
    "-i", input,
    "-vframes", str(frames),
    "-vf", "fps=1",
    "-start_number", "0",
    f"{output}%d.png"
    ])
subprocess.call(command)