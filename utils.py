import subprocess
import ffmpeg

def cudaExists() -> bool:
    """
    Check if CUDA is installed and available on the system.

    :return: True if CUDA is installed and available, False otherwise.
    """
    try:
        subprocess.check_output("nvidia-smi")
        return True
    except Exception:
        return False
    
def getCodec(path: str) -> str:
    """
    Get the codec used in a video file.

    :param: The path to the video file.

    :return: The name of the codec used in the video file.
    """
    return ffmpeg.probe(path)["streams"][0]["codec_name"]