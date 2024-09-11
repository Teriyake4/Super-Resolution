from torch.utils.data import Dataset

from imagequeue import ImageQueue
from utils import ImageTransforms

class SRDataset(Dataset):
    """
    A PyTorch Dataset to be used by a PyTorch DataLoader.
    """
    def __init__(self, queue: ImageQueue, split: str, crop_size: int, scaling_factor: int, lr_img_type: str, hr_img_type: str):
        self.queue = queue
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

        self.queue.startQueue()

        # Select the correct set of transforms
        self.transform = ImageTransforms(split=self.split,
                                         crop_size=self.crop_size,
                                         scaling_factor=self.scaling_factor,
                                         lr_img_type=self.lr_type,
                                         hr_img_type=self.hr_type)
        

    def __getitem__(self, index):
        img = self.queue.get()
        # TODO: REMOVE
        import os
        # print(index)
        # img.save(os.path.join("test media/output/", f"{index}.png"))

        img.convert("RGB")
        if img.width <= 96 or img.height <= 96:
            print(index, img.width, img.height)
        lr_img, hr_img = self.transform(img)

        return lr_img, hr_img

    def __len__(self):
        return self.queue.getTotal()