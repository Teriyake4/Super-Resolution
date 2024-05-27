from collections import deque
import io
import numpy as np
import os
import threading
import time
from typing import List

from .utils import get_codec

import ffmpeg
from PIL import Image
from .imagegroup import ImageGroup


class ImageQueue:
    def __init__(self, queue_size: int, random: bool=True, seed: int=0, use_cuda: bool=False, batch_size: int=1) -> None:
        self.queue_size = queue_size
        self.available_image_groups = []
        # self.image_queue = asyncio.Queue(self.queue_size) # either Image or None
        self.image_queue = deque([], self.queue_size)
        self.batch_size = batch_size
        self.random = random
        self.rng = np.random.default_rng(seed=seed)
        self.use_cuda = use_cuda
        self.queue_busy = threading.Event()
        self.queue_busy.set()

    def addVideos(self, folder_path: str) -> None:
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
            if self.random:
                group.randomize(self.rng)
            new_video_group[i] = group
            i += 1
        self.available_image_groups.extend(new_video_group)

    def addVideo(self, video_path: str) -> None:
        group = ImageGroup(
            video_path,
            int(ffmpeg.probe(video_path)["streams"][0]["nb_frames"]),
            self.batch_size
        )
        if self.random:
            group.randomize(self.rng)
        self.available_image_groups.append(group)
        
    def addImages(self, folder_path: str) -> None:
        image_list = os.listdir(folder_path)
        image_paths = [str] * len(image_list)
        i = 0
        for image in image_list:
            image_paths[i] = os.path.join(folder_path, image)
            i += 1
        group = ImageGroup(image_paths, len(image_list))
        if self.random:
                group.randomize(self.rng)
        self.available_image_groups.append(group)

    def startQueue(self) -> None:
        # asyncio.create_task(self.__init_queue())
        thread = threading.Thread(target=self.__intitQueue)
        thread.start()
        # process = multiprocessing.Process(target=self.__init_queue)
        # process.start()

    def __intitQueue(self) -> None:
        while len(self.available_image_groups) > 0:
            queue_full = self.imagesInQueue() >= self.queue_size
            space_for_batch = self.queue_size - self.imagesInQueue() < self.batch_size
            if queue_full or space_for_batch: # pause adding to queue when queue is full
                time.sleep(0.1)
                continue
            group_index = 0
            if self.random:
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
            print(f"Total: {self.imagesInQueue()}") # debugging

    def __decodeFrames(self, video_path: str, frame_index: List[int]) -> List[Image.Image]: # Error when set output type to List[Image]
        print("DECODING") # debugging
        vcodec = {'vcodec': f"{get_codec(video_path)}_cuvid"} if self.use_cuda else {}
        select = ""
        for i in frame_index:
            select += f"eq(n,{i})+"
        select = select[:-1] # remove last "+" sign
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
        self.queue_busy.wait()
        self.queue_busy.clear()
        if isinstance(image, list):
            self.image_queue.extend(image)
        else:
            self.image_queue.append(image)
        self.queue_busy.set()

    def get(self) -> Image.Image:
        self.queue_busy.wait()
        self.queue_busy.clear()
        image = self.image_queue.popleft()
        self.queue_busy.set()
        return image
    
    def imagesRemaining(self) -> int:
        remaining = self.imagesInQueue()
        for group in self.available_image_groups:
            remaining += group.num_remaining()
        return remaining
    
    def imagesInQueue(self) -> int:
        self.queue_busy.wait()
        self.queue_busy.clear()
        remaining = len(self.image_queue)
        self.queue_busy.set()
        return remaining