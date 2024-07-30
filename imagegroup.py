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

        :param path: video path or list of image file paths.
        :param total_images: Total number of frames in video or total number of images in the directory.
        :param batch_size: Size of each batch to be processed. Defaults to 1.
        """
        self.path = path
        self.total_images = total_images
        self.batch_size = batch_size
        self.available_images = deque(range(self.total_images))

    def randomize(self, rng) -> None:
        """
        Randomizes the order of batches of available images in the ImageGroup.
        
        :param rng: A numpy RandomState object for generating random indices.
        """
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
        self.available_images = deque(temp_array.tolist())
    
    def getPath(self, index: int) -> str:
        """
        Retrieves the file path of an image in the ImageGroup based on its index.

        :param index: The index of the image to retrieve. Must be within the range [0, total_images - 1].
        :return: The file path of the image at the specified index.
        """
        if isinstance(self.path, list):
            return self.path[index]
        return self.path
    
    def getTotalImages(self) -> int:
        """
        Returns the total number of images or frames in the ImageGroup.

        :return: The total number of images or frames in the ImageGroup.
        """
        return self.total_images
    
    def getNext(self) -> List[int]:
        """
        Retrieves the next batch of available images from the ImageGroup.

        :return: A list of indices representing the next batch of available images. If there are no more available images, raises a ValueError.
        """
        if not self.available_images:
            raise ValueError("No more available images to retrieve.")
        next_batch = []
        for _ in range(self.batch_size):
            try:
                next_batch.append(self.available_images.popleft())
            except IndexError:
                break
        return next_batch

    def hasImages(self) -> bool:
        """
        Checks if there are any available images left in the ImageGroup.

        :return: True if there are available images, False otherwise.
        """
        return len(self.available_images) > 0
    
    def numRemaining(self) -> int:
        """
        Returns the number of remaining images or frames in the ImageGroup that have not been retrieved yet.

        :return: The number of remaining images or frames in the ImageGroup.
        """
        return len(self.available_images)
    
    def isVideo(self) -> bool:
        """
        Returns whether the ImageGroup is a group of images or a video.
        
        :return: Returns True if the ImageGroup is a video, False otherwise.
        """
        return not isinstance(self.path, list)
