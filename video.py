"""
Generate video resources for the presentation
"""
import cv2
import carve
import skimage
import masking
import calibrate
import numpy as np
import matplotlib.pyplot as plt
from skimage import transform
from pipeline import get_calibrated_cameras
from config import VIDEOS


IMAGE_WIDTH = 5312
IMAGE_HEIGHT = 2988

def resize(image, desired_width, desired_height):
	return cv2.resize(image, (desired_width, desired_height))


class VideoWriter():

	def __init__(self, filename, cap, N):
		self.filename = filename
		self.cap = cap
		self.N = N
		self.temp = filename.strip('avi')+'png'

		# Calculate the total video shape
		self.width = int(cap[0].get(3))
		self.height = int(cap[0].get(4))
		temp = [np.zeros((self.height, self.width)) for i in range(N)]
		h, w = self.stack(temp).shape

		# Create the video writer
		print("Creating N={0} video with dimensions {1}x{2}".format(N,w,h))
		self.video = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc('M','J','P','G'), 10, (w,h))

	def stack(self,images):
		"""Stack multiple images together"""
		for i in range(len(images)):
			images[i] = resize(images[i], self.width, self.height)
		if self.N==1:
			image = images[0]
		if self.N==2:
			image = np.vstack(images)
		if self.N==4:
			print("dims",[image.shape for image in images])
			im1 = np.hstack(images[:2])
			im2 = np.hstack(images[2:])
			image = np.vstack((im1,im2))
		return image

	def write(self,images):
		"""Write the images to the video and temp file"""
		image = self.stack(images)
		self.video.write(image)
		cv2.imwrite(self.temp, image)

	def release(self):
		"""Release the video"""
		self.video.release()



def test():
	"""
	Test the entire pipeline on a single image
	"""
	model = masking.load_model()
	frame = skimage.io.imread("calibration/note1/raw/15.png")
	cameras = get_calibrated_cameras()

	# Process all frames with mask-rcnn
	result = masking.process_image(model, frame)

	# Extract the mask
	mask = masking.draw_mask_on_image(result, frame)
	cv2.imwrite("output/temp/mask.png", mask)
	print('Mask image shape', mask.shape)

	# Extract all the human silhouettes
	sil = masking.get_human_silhouette(result)

	# Convert the silhoute to an image
	sil_image = masking.convert_sil_to_image(sil)
	cv2.imwrite("output/temp/sil.png", sil_image)
	print('Silhouette image shape', sil_image.shape)

	# Hack to pass dimensions through to carve
	for i in range(len(cameras)):
		cameras[i].image = frame
		cameras[i].silhouette = sil

	# Carve and render the voxels
	silhouettes = [sil for _ in cameras]
	carved = carve.get_carved_image(cameras, silhouettes)
	cv2.imwrite("output/temp/carved.png", carved)


def render_videos():
	# Load the calibrated cameras
	cameras = get_calibrated_cameras()

	# The videos from each camera
	videos = [VIDEOS[c.name] for c in cameras]

	# Load the mask rcnn model
	model = masking.load_model()

	# Load the video objects
	cap = [cv2.VideoCapture(v) for v in videos]

	i = 0
	N = len(cap)

	# Create the output files
	video_raw = VideoWriter("output/videos/raw.avi", cap, N)
	video_mask = VideoWriter("output/videos/mask.avi", cap, N)
	video_sil = VideoWriter("output/videos/sil.avi", cap, N)
	video_carve = VideoWriter("output/videos/carve.avi", cap, 1)
	video_summary = VideoWriter("output/videos/summary.avi", cap, 4)

	# Start iterating through frames
	while all(c.isOpened() for c in cap):
		# Read all the frames
		cap[0].read()
		output = [c.read() for c in cap]
		rets = [i[0] for i in output]
		frames = [i[1] for i in output]
		print(frames[0].shape)

		if not all(rets):
			print("Camera stopped working")
			break

		# Process all frames with mask-rcnn
		results = [masking.process_image(model,f) for f in frames]

		# Extract all of the masks
		masks = [masking.draw_mask_on_image(results[i], frames[i]) for i in range(N)]

		# Extract all the human silhouettes
		sil = [masking.get_human_silhouette(results[i]) for i in range(N)]

		try:
			[s[0] for s in sil]
		except Exception:
			print("Could not find person in every frame")
			continue

		# Compute the OpenCV image versions
		sil_imgs = [masking.convert_sil_to_image(s) for s in sil]

		# Write these for debugging
		video_raw.write(frames)
		video_mask.write(masks)
		video_sil.write(sil_imgs)

		# The carving script is expecting full size images
		sil = [resize(s,IMAGE_WIDTH,IMAGE_HEIGHT) for s in sil]
		print('Sil shape', sil[0].shape)

		# Hack to pass dimensions through to carve
		for i in range(len(cameras)):
			cameras[i].image = frames[i]
			cameras[i].silhouette = sil[i]

		# Carve and render the voxels
		try:
			carved = carve.get_carved_image(cameras, sil)
		except Exception as e:
			print("Voxel carving failed",e)
			continue

		# Write all the videos
		try:
			video_carve.write([carved])
			video_summary.write([frames[0], masks[0], sil_imgs[0], carved])
		except Exception as e:
			print("Video summary failed",e)
			continue


		print("Completed frame %i"%i)
		i+=1

	video_raw.release()
	video_mask.release()
	video_sil.release()
	video_carve.release()
	video_summary.release()
	[c.release() for c in cap]


if __name__=="__main__":
	#test()
	render_videos()