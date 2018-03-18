import os
import cv2
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt

from utils import get_video_writer
from segmentation import coco
from segmentation import utils
from segmentation import model as maskrcnn
from segmentation import visualize


# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "segmentation/logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
	utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
CLASS_NAMES = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
			 'bus', 'train', 'truck', 'boat', 'traffic light',
			 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
			 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
			 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
			 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
			 'kite', 'baseball bat', 'baseball glove', 'skateboard',
			 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
			 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
			 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
			 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
			 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
			 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
			 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
			 'teddy bear', 'hair drier', 'toothbrush']


class InferenceConfig(coco.CocoConfig):
		# Set batch size to 1 since we'll be running inference on
		# one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
		GPU_COUNT = 1
		IMAGES_PER_GPU = 1


def load_model():
	print("Loading MASK R-CNN Model...")
	config = InferenceConfig()
	model = maskrcnn.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
	model.load_weights(COCO_MODEL_PATH, by_name=True)
	print("Done")
	return model


def process_image(model, image):
	"""Process an image with Mask RCNN. Return a results dict"""
	results = model.detect([image], verbose=0)
	return results[0]


def draw_mask_on_image(result, image):
	"""Draw the segmented results on an image
	Return an image (numpy array)
	"""
	r = result
	output = visualize.draw_instances(image, r['rois'], r['masks'], r['class_ids'], CLASS_NAMES, r['scores'])
	cv2.imwrite('output/temp/mask.png',output)
	return output


def get_human_silhouette(result):
	"""
	Return the silhouette that corrosponds to a person
	"""
	person_class = CLASS_NAMES.index('person')
	try:
		person_index = result['class_ids'].tolist().index(person_class)
		return result['masks'][:, :, person_index]
	except ValueError:
		print("Person not found in image")
		return False


def convert_sil_to_image(sil):
	"""Convert sil (binary mask) to a 3-channel BGR image"""
	im = sil[:,:,None]
	return 255*np.tile(im,3)



def draw_mask_on_video(model, input_file, output_file):
	"""
	Draw a mask on every frame and display the video
	"""
	cap = cv2.VideoCapture(input_file)
	out = None

	while(cap.isOpened()):
		print("Reading image from",input_file)
		ret, frame = cap.read()

		print("Drawing masks on frames")
		img = draw_mask_on_frames(model, frame)

		if out is None:
			height = img.shape[0]
			width = img.shape[1]
			out = get_video_writer(output_file, width, height)
			print("Created video writer with size %ix%i"%(width,height))

		print("Writing image to file", output_file)
		out.write(img)

		#print("Displaying last frame")
		#cv2.imshow('frame',im)
		#if cv2.waitKey(1) & 0xFF == ord('q'):
		#			break

	cap.release()
	cv2.destroyAllWindows()








