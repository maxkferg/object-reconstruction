import pickle



def save_object(obj, filename):
	"""Save object to a pickle file"""
	with open(filename, 'wb') as output:  # Overwrites any existing file.
		pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)



def load_object(filename):
	"""Load object from a pickle file"""
	with open(filename, 'rb') as input:
		return pickle.load(input)