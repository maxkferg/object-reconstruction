import os
import cv2
import time
import pickle



def save_object(obj, filename):
	"""Save object to a pickle file"""
	with open(filename, 'wb') as output:  # Overwrites any existing file.
		pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def load_object(filename):
	"""Load object from a pickle file"""
	with open(filename, 'rb') as input:
		return pickle.load(input)


def get_video_writer(filename, width, height):
	"""Return a video writer set to write to filename"""
	# Default resolutions of the frame are obtained.
	frame_width = int(width)
	frame_height = int(height)
	return cv2.VideoWriter(filename,cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))


def record(camera, filename, length):
	"""Record a video from camera"""
	url = camera.url.rsplit("/",1)[0] + "/video"

	print("Recording video on ",camera.name)
	print("Reading from url",url)
	print("Saving video to", filename)

	cap = cv2.VideoCapture(url)

	width = cap.get(3)
	height = cap.get(4)
	out = get_video_writer('output.avi', width, height)

	# Check if camera opened successfully
	if (cap.isOpened() == False):
		raise ValueError("Unable to read camera feed")

	start = time.time()
	duration = time.time() - start

	while duration < length:
		ret, frame = cap.read()
		if ret == True:
			# Write the frame into the file 'output.avi'
			out.write(frame)
			# Display the resulting frame
			cv2.imshow('frame',frame)
			# Press Q on keyboard to stop recording
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
		# Break the loop
		else:
			break
		# Update the running duration
		duration = time.time() - start

	# When everything done, release the video capture and video write objects
	cap.release()
	out.release()

	# Closes all the frames
	cv2.destroyAllWindows()

	return filename