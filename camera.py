import os
import numpy as np
import cv2
import time
import urllib
import requests
import threading
from skimage import transform
from threading import Thread, Event, ThreadError
from config import CAMERAS




class Camera():

	def __init__(self, url, name="default", flip=False):
		self.url = url
		self.name = name
		self.flip = flip


	def __str__(self):
		print(self.name)
		print("mtx:",self.mtx)
		print("dist:",self.dist)
		try:
			print("r:",self.r)
			print("t:",self.t)
		except AttributeError:
			pass
		return self.name


	def set_intrinsics(self, mtx, dist):
		"""Store the camera extrinsics"""
		self.mtx = mtx
		self.dist = dist


	def set_extrinsics(self, r, t):
		"""Store the camera extrinsics"""
		self.r = r
		self.t = t


	def project_points(self, objp):
		"""
		Project world points onto camera image
		inputs:
			-  objp: Object points (3D)
		returns:
			- imgpts: Image points (2D)
		"""
		imgpts, _ = cv2.projectPoints(objp, self.r, self.t, self.mtx, self.dist)
		return imgpts


	def get_camera_direction(self):
		"""Get the unit vector for the direction of the camera"""
		R, _ = cv2.Rodrigues(self.r)
		x = np.array([self.image.shape[1] / 2,
			 self.image.shape[0] / 2,
			 1]);
		X = np.linalg.solve(self.mtx, x)
		X = R.dot(X)
		return X / np.linalg.norm(X)


	def get_image(self, verbose=False):
		"""Return an image from the camera"""
		if verbose:
			print("Getting image from ", self.url)

		# Use urllib to get the image from the IP camera
		imgResp = urllib.request.urlopen(self.url)

		# Numpy to convert into a array
		im = np.array(bytearray(imgResp.read()),dtype=np.uint8)

		im = cv2.imdecode(im,-1)

		# Flip this image if required
		if self.flip:
			im = cv2.flip(im, 0)

		return im


	def show_image(self, verbose=False):
		"""Display an image from the camera"""
		im = self.get_image(verbose)
		im = cv2.resize(im, None, fx=0.5, fy=0.5, interpolation = cv2.INTER_LINEAR)
		cv2.imshow('image',im)
		cv2.waitKey(0)
		cv2.destroyAllWindows()



if __name__=="__main__":
	Camera(CAMERAS[0]).show_image(verbose=True)
	Camera(CAMERAS[1]).show_image(verbose=True)