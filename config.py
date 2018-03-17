
CAMERAS = [
	{
		"url": "http://10.34.177.35:8080/photo.jpg",
		"name": "note1",
		"flip": True
	},
	{
		"url": "http://10.34.177.205:8080/photo.jpg",
		"name": "note2"
	}
]


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


SILHOUETTE = {
	"note1": "photos/note1/silhouette/sil0.png",
	"note2": "photos/note2/silhouette/sil0.png"
}
