import io
import os
import random
import subprocess
from typing import List
import ffmpeg
from PIL import Image
import numpy as np
import torch
# from torchvision.transforms.functional import v2 as FT
import torchvision.transforms.functional as FT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
imgExtentions = [".png", ".heic", ".jpg", ".jpeg"]

# Some constants
rgb_weights = torch.FloatTensor([65.481, 128.553, 24.966]).to(device)
imagenet_mean = torch.FloatTensor([0.485, 0.456, 0.406]).unsqueeze(1).unsqueeze(2)
imagenet_std = torch.FloatTensor([0.229, 0.224, 0.225]).unsqueeze(1).unsqueeze(2)
imagenet_mean_cuda = torch.FloatTensor([0.485, 0.456, 0.406]).to(device).unsqueeze(0).unsqueeze(2).unsqueeze(3)
imagenet_std_cuda = torch.FloatTensor([0.229, 0.224, 0.225]).to(device).unsqueeze(0).unsqueeze(2).unsqueeze(3)


def getDevice() -> str:
    """
    Check for any hardware acceleration, otherwise return cpu
    """
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
    
def getCodec(path: str) -> str:
    """
    Get the codec used in a video file.

    :param: The path to the video file.

    :return: The name of the codec used in the video file.
    """
    return ffmpeg.probe(path)["streams"][0]["codec_name"]

def readFileList(path: str) -> List[str]:
    """
    Read file and return as list where each line is an element.

    :param: The path to the file.

    :return: The list of lines in the file.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r") as file:
        lines = file.readlines()
        file.close()
    return [line.strip() for line in lines]

def extractFrames(path: str) -> List[Image.Image]:
    numFrames = int(ffmpeg.probe(path)["streams"][0]["nb_frames"])
    # out, _ = (
    #     ffmpeg
    #     .input(path)
    #     .output('pipe:', format='rawvideo', pix_fmt='rgb24')
    #     .run(capture_stdout=True, capture_stderr=True)
    # )
    # images = []
    # for i in range(numFrames):
    #     print(i)
    #     frame_data = out[i * 1920 * 1080 * 3:(i + 1) * 1920 * 1080 * 3]
    #     frame = np.frombuffer(frame_data, np.uint8).reshape((1920, 1080, 3))
    #     images.append(Image.fromarray(frame))
    # return images

    # command = [
    #     "ffmpeg",
    #     "-i", path,
    #     "-vf", "fps=1",  # Adjust the fps value as needed
    #     "-f", "image2pipe",
    #     "-vcodec", "png",
    #     "-"
    # ]
    # process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # frames = []
    # while True:
    #     # Read a frame
    #     in_bytes = process.stdout.read(1024 * 1024)
    #     if not in_bytes:
    #         break
    #     # Convert bytes to a PIL image
    #     frame = Image.open(io.BytesIO(in_bytes))
    #     frames.append(frame)
    # return frames

    hwaccel = {}
    device = getDevice()
    if device == "cuda":
        hwaccel = {"vcodec": f"{getCodec(path)}_cuvid"}
    if device == "mps":
        hwaccel = {"hwaccel": "videotoolbox"}
    select = ""
    for i in range(numFrames):
        if i % 1000 == 0:
            select += f"eq(n,{i})+"
    select = select[:-1] # removes last "+" sign
    out, err = (
                ffmpeg
                .input(path, **hwaccel)
                # .filter("filter", select)
                .output("pipe:", vsync="vfr", vframes=numFrames, format="image2pipe", vcodec="png", loglevel="quiet")
                .run(capture_stdout=True) # run_async
            )
    frames = out.split(b"\x89PNG\r\n\x1a\n")
    images = [Image.Image] * numFrames
    i = 0
    for frame in frames:
        if frame:
            image = Image.open(io.BytesIO(b"\x89PNG\r\n\x1a\n" + frame))
            images[i] = image
            i += 1
        print(i)
    # return [Image.open(io.BytesIO(b"\x89PNG\r\n\x1a\n" + frame)) for frame in frames if frame]
    return images


def convert_image(img, source, target):
    """
    Convert an image from a source format to a target format.

    :param img: image
    :param source: source format, one of 'pil' (PIL image), '[0, 1]' or '[-1, 1]' (pixel value ranges)
    :param target: target format, one of 'pil' (PIL image), '[0, 255]', '[0, 1]', '[-1, 1]' (pixel value ranges),
                   'imagenet-norm' (pixel values standardized by imagenet mean and std.),
                   'y-channel' (luminance channel Y in the YCbCr color format, used to calculate PSNR and SSIM)
    :return: converted image
    """
    assert source in {'pil', '[0, 1]', '[-1, 1]'}, "Cannot convert from source format %s!" % source
    assert target in {'pil', '[0, 255]', '[0, 1]', '[-1, 1]', 'imagenet-norm',
                      'y-channel'}, "Cannot convert to target format %s!" % target

    # Convert from source to [0, 1]
    if source == 'pil':
        img = FT.to_tensor(img)

    elif source == '[0, 1]':
        pass  # already in [0, 1]

    elif source == '[-1, 1]':
        img = (img + 1.) / 2.

    # Convert from [0, 1] to target
    if target == 'pil':
        img = FT.to_pil_image(img)

    elif target == '[0, 255]':
        img = 255. * img

    elif target == '[0, 1]':
        pass  # already in [0, 1]

    elif target == '[-1, 1]':
        img = 2. * img - 1.

    elif target == 'imagenet-norm':
        if img.ndimension() == 3:
            img = (img - imagenet_mean) / imagenet_std
        elif img.ndimension() == 4:
            img = (img - imagenet_mean_cuda) / imagenet_std_cuda

    elif target == 'y-channel':
        # Based on definitions at https://github.com/xinntao/BasicSR/wiki/Color-conversion-in-SR
        # torch.dot() does not work the same way as numpy.dot()
        # So, use torch.matmul() to find the dot product between the last dimension of an 4-D tensor and a 1-D tensor
        img = torch.matmul(255. * img.permute(0, 2, 3, 1)[:, 4:-4, 4:-4, :], rgb_weights) / 255. + 16.

    return img


class ImageTransforms(object):
    """
    Image transformation pipeline.
    """

    def __init__(self, split, crop_size, scaling_factor, lr_img_type, hr_img_type):
        """
        :param split: one of 'train' or 'test'
        :param crop_size: crop size of HR images
        :param scaling_factor: LR images will be downsampled from the HR images by this factor
        :param lr_img_type: the target format for the LR image; see convert_image() above for available formats
        :param hr_img_type: the target format for the HR image; see convert_image() above for available formats
        """
        self.split = split.lower()
        self.crop_size = crop_size
        self.scaling_factor = scaling_factor
        self.lr_img_type = lr_img_type
        self.hr_img_type = hr_img_type

        assert self.split in {'train', 'test'}

    def __call__(self, img):
        """
        :param img: a PIL source image from which the HR image will be cropped, and then downsampled to create the LR image
        :return: LR and HR images in the specified format
        """

        # Crop
        if self.split == 'train':
            # Take a random fixed-size crop of the image, which will serve as the high-resolution (HR) image
            left = random.randint(1, img.width - self.crop_size)
            top = random.randint(1, img.height - self.crop_size)
            right = left + self.crop_size
            bottom = top + self.crop_size
            hr_img = img.crop((left, top, right, bottom))
        else:
            # Take the largest possible center-crop of it such that its dimensions are perfectly divisible by the scaling factor
            x_remainder = img.width % self.scaling_factor
            y_remainder = img.height % self.scaling_factor
            left = x_remainder // 2
            top = y_remainder // 2
            right = left + (img.width - x_remainder)
            bottom = top + (img.height - y_remainder)
            hr_img = img.crop((left, top, right, bottom))

        # Downsize this crop to obtain a low-resolution version of it
        lr_img = hr_img.resize((int(hr_img.width / self.scaling_factor), int(hr_img.height / self.scaling_factor)),
                               Image.BICUBIC)

        # Sanity check
        assert hr_img.width == lr_img.width * self.scaling_factor and hr_img.height == lr_img.height * self.scaling_factor

        # Convert the LR and HR image to the required type
        lr_img = convert_image(lr_img, source='pil', target=self.lr_img_type)
        hr_img = convert_image(hr_img, source='pil', target=self.hr_img_type)

        return lr_img, hr_img


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.

    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def save_checkpoint(state, filename):
    """
    Save model checkpoint.

    :param state: checkpoint contents
    """

    torch.save(state, filename)


def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrinks learning rate by a specified factor.

    :param optimizer: optimizer whose learning rate must be shrunk.
    :param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
    """

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))
