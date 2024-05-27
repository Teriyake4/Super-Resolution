from collections import deque
import numpy as np
from typing import List


class ImageGroup:
    """
    Contains either group of images or video to get frames from.
    """
    def __init__(self, path: str | List[str], total_images: int, batch_size: int=1) -> None:
        """
        Initializes an ImageQueue object with specified directory paths or file paths for images.

        :param path: video path or list of image file paths
        :param total_images: Total number of frames in video or total number of images in the directory
        :param batch_size: Size of each batch to be processed. Defaults to 1
        """
        self.path = path
        self.total_images = total_images
        self.batch_size = batch_size
        self.available_images = deque(range(self.total_images))

    def randomize(self, rng) -> None:
        temp_array = np.array(self.available_images)
        if self.batch_size > 1:
            last = temp_array[-(self.numRemaining() % self.batch_size):] # Get the last elements
            temp_array = temp_array[:-(self.numRemaining() % self.batch_size)] # Remove last elemets
            temp_array = np.split(temp_array, (len(temp_array) / self.batch_size)) # Split into chunks
        rng.shuffle(temp_array)
        if self.batch_size > 1:
            temp_array = np.ravel(temp_array) # Combine chunks back into array
            temp_array = np.append(temp_array, last)
        self.available_images = deque(temp_array.tolist())
    
    def getPath(self, index: int) -> str:
        if not isinstance(self.path, list):
            return self.path
        return self.path[index]
    
    def getTotalImages(self) -> int:
        return self.total_images
    
    def getNext(self) -> List[int]:
        # first = self.available_images.popleft()
        next = []
        for i in range(self.batch_size):
            try:
                next.append(self.available_images.popleft())
            except IndexError:
                break
        return next

    def hasImages(self) -> bool:
        return len(self.available_images) > 0
    
    def numRemaining(self) -> int:
        return len(self.available_images)
    
    def isVideo(self) -> bool:
        return not isinstance(self.path, list)