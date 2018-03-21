
CAMERAS = [
	{
		"url": "http://10.34.188.203:8080/photo.jpg",
		"name": "note1",
		"flip": True
	},
	{
		"url": "http://10.34.183.91:8080/photo.jpg",
		"name": "note2"
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
}

# Silhouette images for testing
SILHOUETTE = {
	"note1": "output/note1/silhouette/sil1.png",
	"note2": "output/note2/silhouette/sil1.png"
}

# Raw synchronized videos
VIDEOS = {
	"note1": "output/note1/videos/test2.avi",
	"note2": "output/note2/videos/test2.avi"
}