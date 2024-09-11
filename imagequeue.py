from collections import deque
from concurrent.futures import ThreadPoolExecutor
import io
import sys
import traceback
import numpy as np
import os
import threading
import time
from typing import List

from utils import getCodec

import ffmpeg
from PIL import Image
from imagegroup import ImageGroup

if __name__ == "__main__":
    print(os.environ['PATH'])
    print("In module products sys.path[0], __package__ ==", sys.path[0], __package__)


class ImageQueue:
    """
    The ImageQueue class is a data structure designed to manage and retrieve images from various sources, 
    such as videos or folders of images. 
    """
    def __init__(self, queueSize: int, randomSeed: int=None, batchSize: int=1, device: str="cpu") -> None:
        """
        Constructor for the ImageQueue class.

        :param: queueSize: The maximum number of images that can be stored in the queue at once.
        :param: randomSeed: The seed for whether to shuffle the order of images when retrieving and adding them to the queue. Defaults to None with no shuffling.
        :param: useCuda: Whether to use CUDA for image processing. Defaults to False.
        :param: batchSize: The size of batches to retrieve from the queue at once. Defaults to 1.
        """
        self.queueSize = queueSize
        self.availableImageGroups = []
        # self.image_queue = asyncio.Queue(self.queueSize) # either Image or None
        self.imageQueue = deque([], self.queueSize)
        self.total = 0
        self.batchSize = batchSize
        self.randomSeed = randomSeed
        self.device = device
        self.queueBusy = threading.Event()
        self.queueBusy.set()
        if self.randomSeed is not None:
            self.rng = np.random.default_rng(seed=self.randomSeed)

    def addVideos(self, videoPaths: List[str]) -> None:
        """
        Adds videos from the specified folder to the queue.

        This method takes a folder path as input and adds all the videos in that folder to the queue. 
        Each video is represented by an ImageGroup object.

        :param: folderPath: The path to the folder containing the videos to be added to the queue.
        """
        newVideoGroup = [ImageGroup] * len(videoPaths)
        i = 0
        for video in videoPaths:
            print(video)
            frames = int(ffmpeg.probe(video)["streams"][0]["nb_frames"])
            self.total += frames
            group = ImageGroup(
                video,
                frames,
                self.batchSize
            )
            if self.randomSeed is not None:
                group.randomize(self.rng)
            newVideoGroup[i] = group
            i += 1
        self.availableImageGroups.extend(newVideoGroup)

    def addVideo(self, videoPath: str) -> None:
        """
        Adds a single video to the queue.

        This method takes a video path as input and adds it to the queue. 
        The video is represented by an ImageGroup object.

        :param: videoPath: The path to the video to be added to the queue.
        """
        frames = int(ffmpeg.probe(videoPath)["streams"][0]["nb_frames"])
        self.total += frames
        group = ImageGroup(
            videoPath,
            frames,
            self.batchSize
        )
        if self.randomSeed is not None:
            group.randomize(self.rng)
        self.availableImageGroups.append(group)
    
    def addImages(self, imagePaths: List[str]) -> None:
        """
        Adds multiple images to the queue.

        :param: imagePaths: The list of images' path to be added to the queue.
        """
        self.total += len(imagePaths)
        group = ImageGroup(imagePaths, len(imagePaths))
        if self.randomSeed is not None:
                group.randomize(self.rng)
        self.availableImageGroups.append(group)

    def startQueue(self, endWhenEmpty: bool=True) -> None:
        """
        Starts the queue processing thread.

        This method starts the queue which continuously processes and adds images to the active queue. 
        The queue will stop processing once all images have been processed. 
        """
        thread = threading.Thread(target=self.__intitQueue)
        thread.daemon = True
        thread.start()

    def __intitQueue(self) -> None:
        """
        Initializes the queue processing thread.

        This method continuously processes and adds images to the active queue. 
        It runs indefinitely until explicitly stopped.
        """
        while len(self.availableImageGroups) > 0:
            queue_full = (self.imagesInQueue() >= self.queueSize)
            space_for_batch = self.queueSize - self.imagesInQueue() < self.batchSize
            if queue_full or space_for_batch: # pause adding to queue when queue is full
                time.sleep(0.1)
                continue
            groupIndex = 0
            if self.randomSeed is not None:
                groupIndex = self.rng.integers(low=0, high=len(self.availableImageGroups))
            group = self.availableImageGroups[groupIndex] # select random video/group of images
            if not group.hasImages(): # check if there are any available images
                self.availableImageGroups.pop(groupIndex)
                continue
            imageIndex = group.getNext()
            if group.isVideo():
                images = self.__decodeFrames(group.getPath(imageIndex), imageIndex)
                # if self.random:
                #     temp_array = np.array(images)
                #     self.rng.shuffle(temp_array)
                #     images = temp_array.tolist()
            else:
                try:
                    images = Image.open(group.getPath(imageIndex[0]))
                except Exception as e:
                    print(f"Error opening {group.get_path(imageIndex[0])}")
                    traceback.print_exc()
                    continue
            self.__put(images)

    def __decodeFrames(self, videoPath: str, frameIndex: List[int]) -> List[Image.Image]:
        """
        Decodes frames from the specified video at the given indices.

        This method takes a video path and a list of frame indices as input and returns a list of PIL images 
        representing the decoded frames.

        :param: videoPath: The path to the video from which to decode frames.
        :param: frameIndex: A list of integers representing the indices of the frames to be decoded.
        :return: A list of PIL images representing the decoded frames.
        """
        hwaccel = {}
        if self.device == "cuda":
            hwaccel = {"vcodec": f"{getCodec(videoPath)}_cuvid"}
        elif self.device == "mps":
            hwaccel = {"hwaccel": "videotoolbox"}
        select = ""
        for i in frameIndex:
            select += f"eq(n,{i})+"
        select = select[:-1] # removes last "+" sign
        out, err = (
                ffmpeg
                .input(videoPath, **hwaccel)
                .filter("select", select) # f"gte(n,{frameIndex})"
                .output("pipe:", vsync="vfr", vframes=len(frameIndex), format="image2pipe", vcodec="png", loglevel="quiet")
                .run(capture_stdout=True) # run_async
            )
        frames = out.split(b"\x89PNG\r\n\x1a\n")
        # Convert each frame to a PIL Image
        images = [Image.Image] * len(frameIndex)
        i = 0
        for frame in frames:
            if frame:
                image = Image.open(io.BytesIO(b"\x89PNG\r\n\x1a\n" + frame))
                images[i] = image
                i += 1
        return images
    
    def __put(self, image: Image.Image | List[Image.Image]) -> None:
        """
        Adds an image or a list of images to the queue.

        This method takes either a single PIL image or a list of PIL images as input and adds them to the active queue. 
        The queue is protected by a lock, so only one thread can access it at a time.

        :param: image: A PIL image or a list of PIL images to be added to the queue.
        """
        self.queueBusy.wait()
        self.queueBusy.clear()
        if isinstance(image, list):
            self.imageQueue.extend(image)
        else:
            self.imageQueue.append(image)
        self.queueBusy.set()

    def get(self, wait: bool = True) -> Image.Image | None:
        """
        Retrieves an image from the queue.

        This method retrieves and returns an image from the active queue if it is not empty. If the queue is empty, it returns None.

        :return: An image from the queue or None if the queue is empty.
        """
        if wait:
            while self.imagesInQueue() == 0:
                time.sleep(0.0001)
        self.queueBusy.wait()
        self.queueBusy.clear()
        if len(self.imageQueue) == 0:
            self.queueBusy.set()
            return None
        image = self.imageQueue.popleft()
        self.queueBusy.set()
        return image
    
    def imagesRemaining(self) -> int:
        """
        Returns the total number of remaining images in the queue and images int available image groups.

        :return: The total number of remaining images in the queue and available image groups.
        """
        remaining = self.imagesInQueue()
        for group in self.availableImageGroups:
            remaining += group.numRemaining()
        return remaining
    
    def imagesInQueue(self) -> int:
        """
        Returns the total number of images currently in the queue.

        :return: The total number of images currently in the queue.
        """
        self.queueBusy.wait()
        self.queueBusy.clear()
        remaining = len(self.imageQueue)
        self.queueBusy.set()
        return remaining

    def getTotal(self) -> int:
        return self.total