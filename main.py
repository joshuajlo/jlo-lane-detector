import matplotlib.pyplot as pylot
import numpy as np
import cv2 
import os 
import os
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip
import math


#apply frame masking and finding the region of interest
def region_of_interest(img, vertices):
    if len(img.shape) > 2:
        mask_colour_ignore = (255,) * img.shape[2]
    else:
        mask_colour_ignore = 255

    cv2.fillPoly(np.zeros_like(img), vertices, mask_colour_ignore)
    return cv2.bitwise_and(img, np.zeros_like(img))

#conversion of pixels to a line in hough transform space
def hough_trans(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    lines_drawn(line_img,lines)
    return line_img

#creating the lines for each frame after the trasnform
def lines_drawn(img, lines, color=[255,0,0], thickness = 6):
    global cache
    global first_frame
    slope_1, slope_r = [],[]
    lane_1,lane_r = [], []

    alpha = 0.2

    for line in lines:
        for x1,y1,x2,y2 in line:
            slope = (y2-y1) / (x2-x1)
            if slope > 0.4:
                slope_r.append(slope)
                lane_r.append(line)
            elif slope < -0.4:
                slope_1.append(slope)
                lane_1.append(line)
        img.shape[0] = min(y1,y2,img.shape[0])

    if ((len(lane_1) == 0) or (len(lane_r) ==0)):
        print('No Lane Found')
        return 1
    slope_mean_1 = np.mean(slope_1,axis=0)
    slope_mean_r = np.mean(slope_r,axis=0)
    mean_1 = np.mean(np.array(lane_1),axis=0)
    mean_r = np.mean(np.array(lane_r),axis=0)

    if ((slope_mean_r == 0) or (slope_mean_1 == 0)):
        print('Dividing By Zero')
        return 1


    x1_1 = int((img.shape[0] - mean_1[0][1] - (slope_mean_1 * mean_1[0][0]))/slope_mean_1)
    x2_1 = int((img.shape[0] - mean_1[0][1] - (slope_mean_1 * mean_1[0][0]))/slope_mean_1)
    x1_r = int((img.shape[0] - mean_r[0][1] - (slope_mean_r * mean_r[0][0]))/slope_mean_r)
    x1_r = int((img.shape[0] - mean_r[0][1] - (slope_mean_r * mean_r[0][0]))/slope_mean_r)


    if x1_r > x1_r:
        x1_1 = int((x1_1+x1_r)/2)
        x1_r = x1_1
        y1_1 = int((slope_mean_1 * x1_1 ) + mean_1[0][1] - (slope_mean_1 * mean_1[0][0]))
        y1_r = int((slope_mean_r * x1_r ) + mean_r[0][1] - (slope_mean_r * mean_r[0][0]))
        y2_1 = int((slope_mean_1 * x2_1 ) + mean_1[0][1] - (slope_mean_1 * mean_1[0][0]))
        y2_1 = int((slope_mean_r * x2_r ) + mean_r[0][1] - (slope_mean_r * mean_r[0][0]))

    else:
        y1_1 = img.shape[0]
        y2_1 = img.shape[0]
        y1_r = img.shape[0]
        y2_r = img.shape[0]


    present_frame = np.array([x1_1,y1_1,x2_1,y2_1,x1_r,y1_r,x2_r,y2_r], dtype='float32')

    if first_frame == 1:
        next_frame = present_frame
        first_frame = 0
    else:
        prev_frame = cache
        next_frame = (1-alpha) * prev_frame + alpha*present_frame

    cv2.line(img, (int(next_frame[0]), int(next_frame[1])), (int(next_frame[2]), int(next_frame[3])), color, thickness)
    cv2.line(img, (int(next_frame[4]), int(next_frame[5])), (int(next_frame[6]), int(next_frame[7])), color, thickness)

    cache = next_frame


#process each frame of video to detect lane
def weighted_img(img, initial_img, alpha = 0.8, beta = 1., lda = 0.):
    return cv2.addWeighted(initial_img,alpha,img,beta,lda)


def filter_lines(lines, slope_thresh):
    filtered_lines = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y2 - y1) / (x2 - x1)
            if abs(slope) > slope_thresh:
                filtered_lines.append(line)
    return filtered_lines


def process_image(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)
    
    # Define the region of interest (ROI)
    height, width = edges.shape
    roi_vertices = [(0, height), (width/2, height/2), (width, height)]
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, [np.array(roi_vertices, np.int32)], 255)
    masked_edges = cv2.bitwise_and(edges, mask)
    
    # Apply Hough Line Transform
    lines = cv2.HoughLinesP(masked_edges, 2, np.pi/180, 100, np.array([]), minLineLength=100, maxLineGap=50)
    
    
    if lines is not None:

        filtered_lines = filter_lines(lines, slope_thresh=0.5)

        # Create a blank image to draw lines on
        line_image = np.zeros_like(image)

        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Draw lines in red
    
    # Combine the original image with the line image
        result = cv2.addWeighted(image, 0.8, line_image, 1, 0)
    else:
        result = image

    
    return result

first_frame = 1
white_output = ("/Users/jlo/Projects/jlo-lane-detector/lane-detected.mp4")
clip1 = VideoFileClip("/Users/jlo/Projects/jlo-lane-detector/challenge.mp4")
white_clip = clip1.fl_image(process_image)
white_clip.write_videofile(white_output, audio=False)

