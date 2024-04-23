import ffmpeg
import os

input = os.path.join("test media", "original.mp4")
output = os.path.join("test media", "h264.mp4")
# print(os.path.exists(testvideofile))
# open("output.mp4", "w")

# stream = ffmpeg.input(testvideofile, hwaccel="cuda", hwaccel_device=0)
stream = ffmpeg.input(input, hwaccel="cuda", hwaccel_device=0)
stream = ffmpeg.output(stream, output, vcodec="h264_nvenc", preset="slow", video_bitrate="6000k")
ffmpeg.run(stream)

# vcodec="h264_nvenc"