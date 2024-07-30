from collections import deque
import io
import numpy as np
import os
import threading
import time
from typing import List

from .utils import getCodec

import ffmpeg
from PIL import Image
from .imagegroup import ImageGroup


class ImageQueue:
    """
    The ImageQueue class is a data structure designed to manage and retrieve images from various sources, 
    such as videos or folders of images. 
    """
    def __init__(self, queue_size: int, random_seed: int=None, order: List[str]=None, use_cuda: bool=False, batch_size: int=1) -> None:
        """
        Constructor for the ImageQueue class.

        :param: queue_size: The maximum number of images that can be stored in the queue at once.
        :param: random_seed: The seed for whether to shuffle the order of images when retrieving and adding them to the queue. Defaults to None with no shuffling.
        :param: order: Order in which to retrieve images from
        :param: use_cuda: Whether to use CUDA for image processing. Defaults to False.
        :param: batch_size: The size of batches to retrieve from the queue at once. Defaults to 1.
        """
        self.queue_size = queue_size
        self.available_image_groups = []
        # self.image_queue = asyncio.Queue(self.queue_size) # either Image or None
        self.image_queue = deque([], self.queue_size)
        self.batch_size = batch_size
        self.random_seed = random_seed
        self.order = order
        if self.order is not None:
            assert self.random_seed is None
        self.use_cuda = use_cuda
        self.queue_busy = threading.Event()
        self.queue_busy.set()
        if self.random_seed is not None:
            self.rng = np.random.default_rng(seed=self.random_seed)

    def addVideos(self, folder_path: str) -> None:
        """
        Adds videos from the specified folder to the queue.

        This method takes a folder path as input and adds all the videos in that folder to the queue. 
        Each video is represented by an ImageGroup object.

        :param: folder_path: The path to the folder containing the videos to be added to the queue.
        """
        video_list = os.listdir(folder_path)
        new_video_group = [] * len(video_list)
        i = 0
        for video in video_list:
            path = os.path.join(folder_path, video)
            group = ImageGroup(
                path,
                int(ffmpeg.probe(path)["streams"][0]["nb_frames"]),
                self.batch_size
            )
            if self.random_seed is not None:
                group.randomize(self.rng)
            new_video_group[i] = group
            i += 1
        self.available_image_groups.extend(new_video_group)

    def addVideo(self, video_path: str) -> None:
        """
        Adds a single video to the queue.

        This method takes a video path as input and adds it to the queue. 
        The video is represented by an ImageGroup object.

        :param: video_path: The path to the video to be added to the queue.
        """
        group = ImageGroup(
            video_path,
            int(ffmpeg.probe(video_path)["streams"][0]["nb_frames"]),
            self.batch_size
        )
        if self.random_seed is not None:
            group.randomize(self.rng)
        self.available_image_groups.append(group)
        
    def addImages(self, folder_path: str) -> None:
        """
        Adds images from the specified folder to the queue.

        This method takes a folder path as input and adds all the images in that folder to the queue. 
        Image in the folder are represented by an ImageGroup object.

        :param: folder_path: The path to the folder containing the images to be added to the queue.
        """
        image_list = os.listdir(folder_path)
        image_paths = [str] * len(image_list)
        i = 0
        for image in image_list:
            image_paths[i] = os.path.join(folder_path, image)
            i += 1
        group = ImageGroup(image_paths, len(image_list))
        if self.random_seed is not None:
                group.randomize(self.rng)
        self.available_image_groups.append(group)

    def startQueue(self) -> None:
        """
        Starts the queue processing thread.

        This method creates a new thread that continuously processes and add images to the active queue. 
        The thread runs indefinitely until explicitly stopped.
        """
        # asyncio.create_task(self.__init_queue())
        thread = threading.Thread(target=self.__intitQueue)
        thread.start()
        # process = multiprocessing.Process(target=self.__init_queue)
        # process.start()

    def __intitQueue(self) -> None:
        """
        Initializes the queue processing thread.

        This method continuously processes and adds images to the active queue. 
        It runs indefinitely until explicitly stopped.
        """
        while len(self.available_image_groups) > 0:
            queue_full = self.imagesInQueue() >= self.queue_size
            space_for_batch = self.queue_size - self.imagesInQueue() < self.batch_size
            if queue_full or space_for_batch: # pause adding to queue when queue is full
                time.sleep(0.1)
                continue
            group_index = 0
            if self.random_seed is not None:
                group_index = self.rng.integers(low=0, high=len(self.available_image_groups))
            group = self.available_image_groups[group_index] # select random video/group of images
            if not group.has_images(): # check if there are any available images
                self.available_image_groups.pop(group_index)
                continue
            image_index = group.get_next()
            if group.is_video():
                images = self.__decodeFrames(group.get_path(image_index), image_index)
                # if self.random:
                #     temp_array = np.array(images)
                #     self.rng.shuffle(temp_array)
                #     images = temp_array.tolist()
            else:
                try:
                    images = Image.open(group.get_path(image_index[0]))
                except:
                    print(f"Error opening {group.get_path(image_index[0])}")
                    continue
            self.__put(images)

    def __decodeFrames(self, video_path: str, frame_index: List[int]) -> List[Image.Image]:
        """
        Decodes frames from the specified video at the given indices.

        This method takes a video path and a list of frame indices as input and returns a list of PIL images 
        representing the decoded frames.

        :param: video_path: The path to the video from which to decode frames.
        :param: frame_index: A list of integers representing the indices of the frames to be decoded.
        :return: A list of PIL images representing the decoded frames.
        """
        vcodec = {'vcodec': f"{getCodec(video_path)}_cuvid"} if self.use_cuda else {}
        select = ""
        for i in frame_index:
            select += f"eq(n,{i})+"
        select = select[:-1] # removes last "+" sign
        out, err = (
            ffmpeg
            .input(video_path, **vcodec)
            .filter("select", select) # f"gte(n,{frame_index})"
            .output("pipe:", vsync="vfr", vframes=len(frame_index), format="image2pipe", vcodec="png", loglevel="quiet")
            .run(capture_stdout=True) # run_async
        )
        frames = out.split(b"\x89PNG\r\n\x1a\n")
        # Convert each frame to a PIL Image
        images = [Image.Image] * len(frame_index)
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
        self.queue_busy.wait()
        self.queue_busy.clear()
        if isinstance(image, list):
            self.image_queue.extend(image)
        else:
            self.image_queue.append(image)
        self.queue_busy.set()

    def get(self) -> Image.Image | None:
        """
        Retrieves an image from the queue.

        This method retrieves and returns an image from the active queue if it is not empty. If the queue is empty, it returns None.

        :return: An image from the queue or None if the queue is empty.
        """
        self.queue_busy.wait()
        self.queue_busy.clear()
        if len(self.image_queue) == 0:
            self.queue_busy.set()
            return None
        image = self.image_queue.popleft()
        self.queue_busy.set()
        return image
    
    def imagesRemaining(self) -> int:
        """
        Returns the total number of remaining images in the queue and images int available image groups.

        :return: The total number of remaining images in the queue and available image groups.
        """
        remaining = self.imagesInQueue()
        for group in self.available_image_groups:
            remaining += group.num_remaining()
        return remaining
    
    def imagesInQueue(self) -> int:
        """
        Returns the total number of images currently in the queue.

        :return: The total number of images currently in the queue.
        """
        self.queue_busy.wait()
        self.queue_busy.clear()
        remaining = len(self.image_queue)
        self.queue_busy.set()
        return remaining
