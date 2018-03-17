import os
import glob
import cv2
import time
import urllib
import requests
import threading
import numpy as np
import matplotlib.pyplot as plt
from skimage import color
from threading import Thread, Event, ThreadError
from config import CAMERAS, CALIBRATION, SILHOUETTE
from camera import Camera
from carve import carve
from utils import save_object, load_object
from calibrate import collect_calibration_images, calibrate_camera_intrinsics
from calibrate import take_extrinsic_photo, calibrate_camera_extrinsics
from calibrate import draw_cube_on_chessboard
from calibrate import draw_axis_on_image


def calibrate():
	for camera in CAMERAS:
		camera = Camera(**camera)
		folder = os.path.join("calibration", camera.name)
		folder = os.path.abspath(folder)

		# Taking images
		if not os.path.exists(folder):
			print("Waiting for 60 seconds before sampling images")
			time.sleep(60)
			os.makedirs(folder)
			print("Collecting calibration images for ",camera.name)
			print("Saving images to ", folder)
			collect_calibration_images(camera, folder, n=20)
		print("Intrinsic images have been collected")

		# Calibrating intrinsics
		pickle_file = os.path.join(folder, "intrinsics.pkl")
		if not os.path.exists(pickle_file):
			print("Calibrating intrinsics: ",camera.name)
			camera = calibrate_camera_intrinsics(camera, folder)
			testfile = os.path.join(folder, "raw", "0.png")
			save_object(camera, pickle_file)

		# We load the intrisics from the pickle file
		camera = load_object(pickle_file)
		config = CALIBRATION[camera.name]
		print("Intrinsic parameters have been calculated")

		# Make extrinsic directory
		directory = os.path.dirname(config[0]["file"])
		if not os.path.exists(directory):
			os.makedirs(directory)

		# Take extrinsic photos
		if not os.path.exists(config[0]["file"]):
			take_extrinsic_photo(camera, config[0]["file"])
			take_extrinsic_photo(camera, config[1]["file"])
		print("Extrinsic images have been collected")

		# Use the extrinsic photos to calibrate
		extrinsic_file = os.path.join(folder, "extrinsics.pkl")
		if not os.path.exists(extrinsic_file):
			print("Calibrating extrinsics: ",camera.name)
			camera = calibrate_camera_extrinsics(camera, config)
			save_object(camera, extrinsic_file)
			cv2.destroyAllWindows()
		print("Extrinsic parameters have been calculated")

		# Show camera extrinsics
		print("Loading extrinsics from",extrinsic_file)
		camera = load_object(extrinsic_file)

		# Make tests directory
		test_directory = os.path.join(folder, "tests")
		if not os.path.exists(test_directory):
			print("Making test directory")
			os.makedirs(test_directory)

		print("Creating test images")
		draw_axis_on_image(camera, config[0]["file"], os.path.join(test_directory,"test0.png"))
		draw_axis_on_image(camera, config[1]["file"], os.path.join(test_directory,"test8.png"))



def pipeline():
	cameras = []
	silhouttes = []
	for config in CAMERAS:
		folder = os.path.join("calibration", config["name"])
		extrinsic_file = os.path.join(folder, "extrinsics.pkl")
		camera = load_object(extrinsic_file)
		cameras.append(camera)
		# Open the silhoutte
		silhouette_file = SILHOUETTE[camera.name]
		silhouette_img = plt.imread(silhouette_file)
		silhouette_img = color.rgb2gray(silhouette_img)
		silhouttes.append(silhouette_img)
		camera.image = silhouette_img # HACK
		camera.silhouette = silhouette_img # HACK
	# Carve the voxels
	carve(cameras,silhouttes)



if __name__=="__main__":
	calibrate()
	pipeline()