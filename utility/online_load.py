import requests
import numpy as np
from PIL import Image, ImageSequence
import os

def load_gif(url:str,keep_gif:bool = False) -> np.array :

    #retrieving gif from URL
    with open("tmp.gif", 'wb') as f:
        f.write(requests.get(url).content)

    #converting PIL to numpy
    with Image.open("tmp.gif") as img_array:
        frames = []
        for frame in ImageSequence.Iterator(img_array):
            #L: grayscale
            #RGB: colored
            frame = frame.convert("L") 
            frames.append(np.array(frame))

        if keep_gif == False:
            os.remove("tmp.gif")
        return np.array(frames)

def main():
    url = 'https://31.media.tumblr.com/00110d6b41d354747240a9308ae522bb/tumblr_n9n04znLpA1qddk8uo1_500.gif'
    df = load_gif(url)
    print(f"Frames: {df.shape[0]}")
    print(f"Resolution: {df.shape[2]}x{df.shape[1]}")

if __name__ == "__main__":
    main()

