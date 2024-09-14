import subprocess
import time

for i in range(150):
    start = time.time()
    # 28 sec
    command = [
        "ffmpeg",
        "-noaccurate_seek",
        "-ss", f"{i}",
        "-i", "test media/Valorant 2022.06.02 - 14.30.39.03.DVR.mp4",
        "-vframes", "1",
        f"test media/output/{i}.png"
    ]

    # VERY SLOW
    # command = [
    #     "ffmpeg",
    #     "-i", "test media/Valorant 2022.06.02 - 14.30.39.03.DVR.mp4",
    #     "-vf", f"select=eq(n\,{i*60})",
    #     "-vsync", "vfr",
    #     "-q:v", "2",
    #     "-vframes", "1",
    #     f"test media/output/{i}.png"
    # ]

    # subprocess.call(command)
    subprocess.call(command, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    print(f"Time: {time.time() - start}")
