import os
import glob
import cv2
import time
import urllib
import requests
import threading
import masking
import numpy as np
import matplotlib.pyplot as plt
from skimage import color
from threading import Thread, Event, ThreadError
from config import CAMERAS, CALIBRATION, SILHOUETTE, VIDEOS
from camera import Camera
from carve import carve
from utils import save_object, load_object, record
from calibrate import collect_calibration_images, calibrate_camera_intrinsics
from calibrate import take_extrinsic_photo, calibrate_camera_extrinsics
from calibrate import draw_cube_on_chessboard
from calibrate import draw_axis_on_image


def get_calibrated_cameras():
	"""Return a list of calibrated camera objects"""
	cameras = []
	for config in CAMERAS:
		folder = os.path.join("calibration", config["name"])
		extrinsic_file = os.path.join(folder, "extrinsics.pkl")
		camera = load_object(extrinsic_file)
		camera.url = config["url"]
		cameras.append(camera)
	return cameras


def calibrate():
	for config in CAMERAS:
		camera = Camera(**config)
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
		camera.url = config["url"]
		calib = CALIBRATION[camera.name]
		print("Intrinsic parameters have been calculated")

		# Make extrinsic directory
		directory = os.path.dirname(calib[0]["file"])
		if not os.path.exists(directory):
			os.makedirs(directory)

		# Take extrinsic photos
		if not os.path.exists(calib[0]["file"]):
			take_extrinsic_photo(camera, calib[0]["file"])
			take_extrinsic_photo(camera, calib[1]["file"])
		print("Extrinsic images have been collected")

		# Use the extrinsic photos to calibrate
		extrinsic_file = os.path.join(folder, "extrinsics.pkl")
		if not os.path.exists(extrinsic_file):
			print("Calibrating extrinsics: ",camera.name)
			camera = calibrate_camera_extrinsics(camera, calib)
			save_object(camera, extrinsic_file)
			cv2.destroyAllWindows()
		print("Extrinsic parameters have been calculated")

		# Show camera extrinsics
		print("Loading extrinsics from",extrinsic_file)
		camera = load_object(extrinsic_file)
		camera.url = config["url"]

		# Make tests directory
		test_directory = os.path.join(folder, "tests")
		if not os.path.exists(test_directory):
			print("Making test directory")
			os.makedirs(test_directory)

		print("Creating test images")
		draw_axis_on_image(camera, calib[0]["file"], os.path.join(test_directory,"test0.png"))
		draw_axis_on_image(camera, calib[1]["file"], os.path.join(test_directory,"test8.png"))


def carving():
	cameras = []
	silhouttes = []
	cameras = get_calibrated_cameras()
	for camera in cameras:
		# Open the silhoutte
		silhouette_file = SILHOUETTE[camera.name]
		silhouette_img = plt.imread(silhouette_file)
		silhouette_img = color.rgb2gray(silhouette_img)
		silhouttes.append(silhouette_img)
		camera.image = silhouette_img # HACK
		camera.silhouette = silhouette_img # HACK
	# Carve the voxels
	carve(cameras,silhouttes)



def record_on_all_cameras():
	"""
	Record a scene
	Records on every camera
	"""
	cameras = get_calibrated_cameras()
	length = 10

	# Define a lambda recording function
	def start_recording(camera):
		print("Thread %i started"%i)
		record(camera, VIDEOS[camera.name], length)
		print("Thread %i completed"%i)

	pool = multiprocessing.Pool(processes=n)
	pool.map(start_recording, cameras)
	pool.close()
	pool.join()
	print('Finished recording!')






if __name__=="__main__":
	#calibrate()
	carving()