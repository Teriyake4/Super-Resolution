import os
from imagequeue import ImageQueue
import utils
import time


QUEUE_SIZE = 2048
RANDOM_SEED = None
DEVICE = utils.getDevice()
BATCH_SIZE = 128

video_path = "test media/Valorant 2022.06.02 - 14.30.39.03.DVR.mp4"
output_path = "test media/output/"

queue = ImageQueue(QUEUE_SIZE, RANDOM_SEED, BATCH_SIZE, DEVICE)
queue.addVideo(video_path)
print("Starting queue")
queue.startQueue()

i = 0
while queue.imagesRemaining() > 0:
    if queue.imagesInQueue() == 0:
        continue
    print(f"Getting image {i}")
    image = queue.get()
    if i % 1000 == 0:
        image.save(os.path.join(output_path, f"{i}.png"))
        # print(f"Saving {i}" )
    i += 1
    # time.sleep(1)

print("Finished queue")