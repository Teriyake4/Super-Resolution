import io
import subprocess
import ffmpeg
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

from utils import ImageTransforms
import utils

class SRDataset(Dataset):
    """
    A PyTorch Dataset to be used by a PyTorch DataLoader.
    """
    def __init__(self, video_path: str, device: str, split: str, crop_size: int, scaling_factor: int, lr_img_type: str, hr_img_type: str):
        self.video_path = video_path
        self.device = device
        self.split = split
        self.crop_size = crop_size
        self.scaling_factor = scaling_factor
        self.lr_type = lr_img_type
        self.hr_type = hr_img_type

        assert self.split in {"train", "test"}
        assert self.lr_type in {'[0, 255]', '[0, 1]', '[-1, 1]', 'imagenet-norm'}
        assert self.hr_type in {'[0, 255]', '[0, 1]', '[-1, 1]', 'imagenet-norm'}

        if self.split == 'train':
            assert self.crop_size % self.scaling_factor == 0, "Crop dimensions are not perfectly divisible by scaling factor! This will lead to a mismatch in the dimensions of the original HR patches and their super-resolved (SR) versions!"

        # self.images = utils.extractFrames(self.video_path)
        self.images = [None] * self.__len__()
        self.codec = utils.getCodec(self.video_path)
        # Select the correct set of transforms
        self.transform = ImageTransforms(split=self.split,
                                         crop_size=self.crop_size,
                                         scaling_factor=self.scaling_factor,
                                         lr_img_type=self.lr_type,
                                         hr_img_type=self.hr_type)
        

    def __getitem__(self, index):
        if self.images[index] is None:
            hwaccel = {}
            if self.device == "cuda":
                hwaccel = {"hwaccel": "cuda"}
            elif self.device == "mps":
                hwaccel = {"hwaccel": "videotoolbox"}
            try:
                out, err = ( # 27854
                    ffmpeg
                    .input(self.video_path, ss=index, **hwaccel) # noaccurate_seek=None,
                    .output('pipe:', vframes=1, format='image2', vcodec='png') # loglevel="quiet"
                    .run(capture_stdout=True, capture_stderr=True)
                )
            except ffmpeg.Error as e:
                print(f"Index: : {index}")
                print('stdout:', e.stdout.decode('utf8'))
                print('stderr:', e.stderr.decode('utf8'))
                print(f"err {e.stderr}")
                raise e
            self.images[index] = Image.open(io.BytesIO(out))
        img = self.images[index]
        # print(f"Index: : {index}")

        # TODO: REMOVE
        import os
        # img.save(os.path.join("test media/output/", f"{index}.png"))
        # print(f"Got {index}")
        img.convert("RGB")
        if img.width <= 96 or img.height <= 96:
            print(index, img.width, img.height)
        lr_img, hr_img = self.transform(img)

        return lr_img, hr_img

    def __len__(self):
        # return len(self.images)
        return int(ffmpeg.probe(self.video_path)["streams"][0]["nb_frames"])