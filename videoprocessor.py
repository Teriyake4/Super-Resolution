import subprocess
import ffmpeg
import json
from PIL import Image
import io

def _cudaexists() -> bool:
    try:
        subprocess.check_output("nvidia-smi")
        return True
    except Exception:
        return False

def _getvcodec(meta) -> str:
    return meta["codec_name"]

def _getvbitrate(meta, x_res: int=None) -> int:
    multiplier = 1
    if x_res is not None:
        orignialres = meta["coded_width"]
        if x_res > orignialres:
            multiplier = (x_res / orignialres) ** 2
        else:
            multiplier = (orignialres / x_res) ** 2
    return meta["bit_rate"] * multiplier

def transcode(input: str, output: str, preset: str, aspipe: bool=False, vcodec: str=None, vbitrate: str=None, x_res: int=None):
    stream = None
    meta = None
    scale = "scale"
    # If no video codec is specified, fetch the one from input file"s metadata
    if vcodec is None:
        meta = ffmpeg.probe(input)["streams"][0]
        vcodec = _getvcodec(meta)
    # If no video bitrate is specified, fetch it from input file"s metadata
    if vbitrate is None:
        if meta is None:
            meta = ffmpeg.probe(input)["streams"][0]
        vbitrate = _getvbitrate(meta)
    # If CUDA exists, use it for encoding and set the codec as a CUDA-compatible one
    if _cudaexists():
        vcodec = vcodec + "_nvenc"
        scale = "scale_cuda"
        stream = ffmpeg.input(input, hwaccel="cuda", hwaccel_output_format="cuda")
    if x_res != None:
        stream = ffmpeg.filter(stream, scale, x_res, -2)
    # if aspipe:
        # ffmpeg.output(stream, "pipe:", format="rawvideo", pix_fmt="rgb24", preset=preset, vcodec=vcodec, acodec="copy", video_bitrate=vbitrate, audio_bitrate="copy")
    
    output = "pipe:" if aspipe else output
    format = "rawvideo" if aspipe else None
    pix_fmt = "rgb24" if aspipe else None

    # Configure the output settings for the video stream
    output_settings = {
        # "format": format,
        # "pix_fmt": pix_fmt,
        "preset": preset,
        "vcodec": vcodec,
        "acodec": "copy",  # Copy audio codec without re-encoding
        "video_bitrate": vbitrate,
        # "audio_bitrate": "copy"  # Copy audio bitrate without re-encoding
    }
    stream = ffmpeg.output(stream, output, **output_settings)
    if True:
        print(ffmpeg.compile(stream))
        for i in ffmpeg.compile(stream):
            print(f"{i} ", end="")
    pipe = ffmpeg.run(stream)[0]
    if aspipe:
        return pipe

def videotoimg(input: str, output: str):
    out, err = (
        ffmpeg
        .input(input, vcodec="h264_cuvid")
        .output(output, format='image2', vcodec="png")
        .run()
    )
    return out

def framefromvideo(input: str, frame: int = 0):
    # return (
    #     ffmpeg
    #     .input(input)
    #     .filter("select", f"gte(n, {frame})")
    #     .output(vframe=frame)
    # )
    out, err = (
        ffmpeg
        .input(input, vcodec="h264_cuvid") # , vcodec="h264_cuvid"
        .filter("select", f"gte(n,{frame})")
        .output("pipe:", vframes=1, format="image2", vcodec="png")
        .run(capture_stdout=True) # run_async
    )
    return out

def framesfromvideo(input: str, frame: int = 0, num: int = 5):
    # return (
    #     ffmpeg
    #     .input(input)
    #     .filter("select", f"gte(n, {frame})")
    #     .output(vframe=frame)
    # )
    out, err = (
        ffmpeg
        .input(input, vcodec="h264") # , vcodec="h264_cuvid"
        .filter("select", f"gte(n,{frame})")
        .output("pipe:", vsync="vfr", vframes=num, format="image2pipe", vcodec="png")
        .run(capture_stdout=True) # run_async
    )
    frames = out.split(b'\x89PNG\r\n\x1a\n')
    print(len(frames))
    # Convert each frame to a PIL Image
    images = [Image] * len(frames)
    i = 0
    for frame in frames:
        if frame:
            print(i)
            image = Image.open(io.BytesIO(b'\x89PNG\r\n\x1a\n' + frame))
            images[i] = image
            i += 1
    return images

def frames(input: str, output: str, frame: int, num: int = 5):
    (
        ffmpeg
        .input(input)
        .filter("select", f"gte(n,{frame})") # or eq, between
        .output(output, vsync="vfr", vframes=num)
        .run()
    )
