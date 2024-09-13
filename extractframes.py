import io
import os
from typing import List
from PIL import Image

import ffmpeg

import utils


input = "test media/out.mp4"
output = "/test media/output/"

framerate = 1
device = utils.getDevice()

# List all videos and images as txt in random order
# breaking up videos?
# ffmpeg -hwaccel videotoolbox -i "Valorant 2022.06.02 - 14.30.39.03.DVR.mp4" -vf fps=1 frame_%04d.png
# frames = int(ffmpeg.probe(input)["streams"][0]["nb_frames"])
# hwaccel = {}
# if device == "cuda":
#     hwaccel = {"hwaccel": "cuvid"}
# elif device == "mps":
#     hwaccel = {"hwaccel": "videotoolbox"}
numFrames = int(ffmpeg.probe(input)["streams"][0]["nb_frames"])
hwaccel = {}
device = utils.getDevice()
if device == "cuda":
    hwaccel = {"vcodec": f"{utils.etCodec(input)}_cuvid"}
if device == "mps":
    hwaccel = {"hwaccel": "videotoolbox"}
out, err = (
    ffmpeg
    .input(input, **hwaccel)
    # .filter("filter", select)
    .output("pipe:", vsync="vfr", vframes=numFrames, format="image2pipe", vcodec="png") # loglevel="quiet"
    .run(capture_stdout=True) # run_async
)
print(err)
frames = out.split(b"\x89PNG\r\n\x1a\n")
i = 0
for frame in frames:
    if frame:
        i += 1
        image = Image.open(io.BytesIO(b"\x89PNG\r\n\x1a\n" + frame))
        image.save(f"{output}{i}.png")