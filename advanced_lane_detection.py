import matplotlib.pyplot as plt
import cv2
import imageio
imageio.plugins.ffmpeg.download()
from moviepy.editor import VideoFileClip

from camera import camera
from lane_detector import LaneDetector

image = plt.imread("test_images/test2.jpg")
ld = LaneDetector(image)
#plt.imshow(ld.find_curvature_sequence(image))

output = 'project_video_output.mp4'
input_clip = VideoFileClip("project_video.mp4")
output_clip = input_clip.fl_image(ld.find_curvature_sequence)
output_clip.write_videofile(output, audio=False)
