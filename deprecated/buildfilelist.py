from concurrent.futures import ThreadPoolExecutor
import os
from typing import List
from PIL import Image

import utils


pathList = [""]
fileList = "filelist.txt"

# List all videos and images as txt in random order
# breaking up videos?
def checkFileValidity(filePath: str) -> bool:
    if not os.path.exists(filePath):
        return False
    root, extension = os.path.splitext(filePath)
    if extension == ".mp4":
        return True
    if extension not in utils.imgExtentions:
        return False
    try:
        with Image.open(filePath) as img:
            img.verify()
    except Exception as e:
        return False
    return True

def getFiles(path: str) -> List[str]:
    fileList = []
    for root, dirs, files in os.walk(path):
        dirs = [os.path.join(root, d) for d in dirs]
        files = [os.path.join(root, f) for f in files if checkFileValidity(os.path.join(root, f))]
        fileList.extend(files)
    return fileList


sourceList = []
for path in pathList:
    if not os.path.exists(path):
        continue
    sourceList = getFiles(path)
print("Finished reading")

last = sourceList[-1]
sourceList = [str(source) + "\n" for source in sourceList[0:len(sourceList)-1]]
sourceList.append(last)

print(f"Adding {len(sourceList)} items")
with open(fileList, "w") as file:
    file.writelines(sourceList)
    file.close()
