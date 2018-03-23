
CAMERAS = [
	{
		"url": "http://10.34.177.35:8080/photo.jpg",
		"name": "note1",
		"flip": True
	},
	{
		"url": "http://10.34.177.205:8080/photo.jpg",
		"name": "note2"
	},
	{
		"url": "http://10.34.178.105:8080/photo.jpg",
		"name": "note3"
	}
]


# Extrinsic calibration
CALIBRATION = {
	"note1" : [
		{
			"z":0,
		 	"file": "calibration/note1/extrinsics/0.png",
		},
		{
			"z":8,
		 	"file": "calibration/note1/extrinsics/8.png",
		}

	],
	"note2": [
		{
			"z":0,
		 	"file": "calibration/note2/extrinsics/0.png",
		},
		{
			"z":8,
		 	"file": "calibration/note2/extrinsics/8.png",
		}
	],
	"note3": [
		{
			"z":0,
		 	"file": "calibration/note3/extrinsics/0.png",
		},
		{
			"z":8,
		 	"file": "calibration/note3/extrinsics/8.png",
		}
	],
}

# Silhouette images for testing
SILHOUETTE = {
	"note1": "output/note1/silhouette/sil1.png",
	"note2": "output/note2/silhouette/sil2.png"
	"note3": "output/note3/silhouette/sil2.png"
}


OBJECTS = {
	"person": {
		"note1": "output/note1/silhouette/sil-person-note1.png",
		"note2": "output/note2/silhouette/sil-person-note2.png"
		"note3": "output/note3/silhouette/sil-person-note3.png"
	},
	"chair": {
		"note1": "output/note1/silhouette/sil-chair-note1.png",
		"note2": "output/note2/silhouette/sil-chair-note2.png"
		"note3": "output/note2/silhouette/sil-chair-note3.png"
	}
}

# Load videos from here for pipeline
VIDEOS_IN = {
	"note1": "output/note1/videos/test2.avi",
	"note2": "output/note2/videos/test2.avi"
	"note3": "output/note3/videos/test2.avi"
}

# Write results to here
RESULTS_DIR = "output/videos/test"

# Raw synchronized videos
VIDEOS_OUT = {
	"note1": "output/note1/videos/test3.avi",
	"note2": "output/note2/videos/test3.avi"
	"note3": "output/note3/videos/test3.avi"
}