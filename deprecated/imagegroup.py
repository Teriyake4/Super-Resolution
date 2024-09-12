from collections import deque
import numpy as np
from typing import List


class ImageGroup:
    """
    Contains either group of images or video to get frames from.
    """
    def __init__(self, path: str | List[str], totalImages: int, batchSize: int=1) -> None:
        """
        Initializes an ImageQueue object with specified directory paths or file paths for images.

        :param path: video path or list of image file paths.
        :param totalImages: Total number of frames in video or total number of images in the directory.
        :param batchSize: Size of each batch to be processed. Defaults to 1.
        """
        self.path = path
        self.totalImages = totalImages
        self.batchSize = batchSize
        self.availableImages = deque(range(self.totalImages))

    def randomize(self, rng) -> None:
        """
        Randomizes the order of batches of available images in the ImageGroup.
        
        :param rng: A numpy RandomState object for generating random indices.
        """
        tempArray = np.array(self.availableImages)
        if self.batchSize > 1:
            last = tempArray[-(self.numRemaining() % self.batchSize):] # Get the last elements
            tempArray = tempArray[:-(self.numRemaining() % self.batchSize)] # Remove last elemets
            tempArray = np.split(tempArray, (len(tempArray) / self.batchSize)) # Split into chunks
        rng.shuffle(tempArray)
        if self.batchSize > 1:
            tempArray = np.ravel(tempArray) # Combine chunks back into array
            tempArray = np.append(tempArray, last)
        self.availableImages = deque(tempArray.tolist())
        self.availableImages = deque(tempArray.tolist())
    
    def getPath(self, index: int) -> str:
        """
        Retrieves the file path of an image in the ImageGroup based on its index.

        :param index: The index of the image to retrieve. Must be within the range [0, totalImages - 1].
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
        return self.totalImages
    
    def getNext(self) -> List[int]:
        """
        Retrieves the next batch of available images from the ImageGroup.

        :return: A list of indices representing the next batch of available images. If there are no more available images, raises a ValueError.
        """
        if not self.availableImages:
            raise ValueError("No more available images to retrieve.")
        next_batch = []
        for _ in range(self.batchSize):
            try:
                next_batch.append(self.availableImages.popleft())
            except IndexError:
                break
        # TODO remove later
        self.availableImages.extend(next_batch)
        return next_batch

    def hasImages(self) -> bool:
        """
        Checks if there are any available images left in the ImageGroup.

        :return: True if there are available images, False otherwise.
        """
        return len(self.availableImages) > 0
    
    def numRemaining(self) -> int:
        """
        Returns the number of remaining images or frames in the ImageGroup that have not been retrieved yet.

        :return: The number of remaining images or frames in the ImageGroup.
        """
        return len(self.availableImages)
    
    def isVideo(self) -> bool:
        """
        Returns whether the ImageGroup is a group of images or a video.
        
        :return: Returns True if the ImageGroup is a video, False otherwise.
        """
        return not isinstance(self.path, list)
