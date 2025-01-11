
import sys
import os
sys.path.append(os.getcwd())
from utility.online_load import load_gif
import cv2
import pygame
import time


def test_load_gif(keep_gif:bool = False) -> None:
    url = 'https://31.media.tumblr.com/00110d6b41d354747240a9308ae522bb/tumblr_n9n04znLpA1qddk8uo1_500.gif'
    df = load_gif(url,True)
    print(f"Frames: {df.shape[0]}")
    print(f"Resolution: {df.shape[2]}x{df.shape[1]}")
    cap = cv2.VideoCapture("tmp.gif")
    while(1):
        ret, frame = cap.read()
        if cv2.waitKey(1) & 0xFF == ord('q') or ret==False :
            cap.release()
            cv2.destroyAllWindows()
            break
        cv2.imshow('frame',frame)
        time.sleep(0.05)
    os.remove("tmp.gif")

if __name__ == "__main__":
    test_load_gif()