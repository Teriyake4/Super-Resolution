import subprocess
import ffmpeg

def cuda_exists() -> bool:
    try:
        subprocess.check_output("nvidia-smi")
        return True
    except Exception:
        return False
    
def get_codec(path: str) -> str:
    return ffmpeg.probe(path)["streams"][0]["codec_name"]