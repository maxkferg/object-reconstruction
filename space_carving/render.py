# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) Vispy Development Team. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
# -----------------------------------------------------------------------------

"""
This example demonstrates isocurve for triangular mesh with vertex data.
"""

import numpy as np
from vispy import app, scene
from vispy.geometry.generation import create_sphere
from vispy.visuals.transforms import MatrixTransform
from vispy import app, gloo, visuals
import sys



def render(vertices, normals, faces):

	# Create a canvas with a 3D viewport
	canvas = scene.SceneCanvas(keys='interactive',
	                           title='Isocurve for Triangular Mesh Example')

	canvas.show()
	view = canvas.central_widget.add_view()
	zaxis = -1.0

	# Draw the meshgrid
	vertices = vertices/60
	nv = vertices.size//3
	vcolor = np.ones((nv, 4), dtype=np.float32)
	vcolor[:, 0] = np.linspace(1, 0, nv)
	vcolor[:, 1] = 0.2#np.linspace(1, 0, nv)
	vcolor[:, 2] = np.linspace(0, 1, nv)
	mesh = scene.visuals.Mesh(vertices=vertices, faces=faces, vertex_colors=vcolor, shading='smooth', parent=view.scene)
	tr = MatrixTransform()
	tr.rotate(-90, (1, 0, 0))
	tr.translate((0, 0, 1.2))
	mesh.transform = tr
	mesh.shininess = 0.001

	# Add a 3D axis to keep us oriented
	ax = scene.visuals.XYZAxis(parent=view.scene)
	tr = MatrixTransform()
	tr.translate((0, 0, zaxis))
	ax.transform = tr

	view.camera = scene.TurntableCamera(fov=40, elevation=14.0, distance=8)
	view.camera.azimuth = -45

	# Draw the chessboard
	# Build a checkerboard colored square plane with "segments" number of tiles per side.
	size = 0.5
	segments = 8
	white = (0.8, 0.8, 0.98, 0.8)
	black = (0.26, 0.26, 0.39, 0.8)
	materials = [white, black]

	for x in range(-1,segments):
		for y in range(-1,segments):
			i = x * segments + y
			j = 2 * i
			color = materials[(x+y) % 2]
			position = size * np.array([[x,y,0], [x+1,y,0], [x+1,y+1,0], [x,y+1,0]])
			polygon = scene.visuals.Polygon(position, color=color, parent=view.scene)
			polygon.transform = tr

	return app, canvas

