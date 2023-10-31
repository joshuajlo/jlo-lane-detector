import tkinter as tk
from tkinter import *
import cv2
from PIL import Image, ImageTk
import os
import numpy as np


global last_frame1
last_frame = np.zeros((480, 640, 3), dtype=np.uint8)
global last_frame2
last_frame2 = np.zeros((490, 640, 3), dtype=np.uint8)
global cap1, cap2
cap1 = cv2.VideoCapture("//Users/jlo/Projects/jlo-lane-detector/challenge.mp4")
cap2 = cv2.VideoCapture("/Users/jlo/Projects/jlo-lane-detector/lane-detected.mp4")


def show_vid():
	if not cap1.isOpened():
		print('Cannot Open Camera 1')
	flag1, frame1 = cap1.read()
	frame1 = cv2.resize(frame1, (400,500))
	if flag1 is None:
		print('Error')
	elif flag1:
		global last_frame1
		last_frame1 = frame1.copy()
		pic = cv2.cvtColor(last_frame1, cv2.COLOR_BGR2RGB)
		img = Image.fromarray(pic)
		imgtk = ImageTk.PhotoImage(image=img)
		lmain.imgtk = imgtk
		lmain.configure(image=imgtk)
		lmain.after(10, show_vid)

def show_vid2():
	if not cap2.isOpened():
		print('Cannot Open Camera 2')
	flag2, frame2 = cap2.read()
	frame2 = cv2.resize(frame2, (400,500))
	if flag2 is None:
		print('Error')
	elif flag2:
		global last_frame2
		last_frame2 = frame2.copy()
		pic = cv2.cvtColor(last_frame2, cv2.COLOR_BGR2RGB)
		img2 = Image.fromarray(pic)
		img2tk = ImageTk.PhotoImage(image=img2)
		lmain2.img2tk = img2tk
		lmain2.configure(image=img2tk)
		lmain2.after(10, show_vid2)



if __name__ == '__main__':
	while True:
		root=tk.Tk()
		lmain = tk.Label(master=root)
		lmain2 = tk.Label(master=root)

		lmain.pack(side = LEFT)
		lmain2.pack(side = RIGHT)
		root.title('JLo Lane Detector')
		root.geometry('900x700+100+10')
		exit = Button(root, text = 'Exit', fg='red',command = root.destroy).pack(side = BOTTOM,)
		show_vid()
		show_vid2()
		root.mainloop()
	cap.release()