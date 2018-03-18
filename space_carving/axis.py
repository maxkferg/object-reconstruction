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

	# Draw the meshgrid
	vertices = vertices/20
	nv = vertices.size//3
	vcolor = np.ones((nv, 4), dtype=np.float32)
	vcolor[:, 0] = np.linspace(1, 0, nv)
	vcolor[:, 1] = 0.2#np.linspace(1, 0, nv)
	vcolor[:, 2] = np.linspace(0, 1, nv)
	mesh = scene.visuals.Mesh(vertices=vertices, faces=faces, vertex_colors=vcolor, shading='smooth', parent=view.scene)
	tr = MatrixTransform()
	tr.rotate(-90, (1, 0, 0))
	tr.translate((0, 0, 0.8))
	mesh.transform = tr
	#mesh.shininess = 0.01
	#mesh.cmap = 'viridis'

	# Draw the circles
	cols = 10
	rows = 10
	radius = 2
	nbr_level = 20
	mesh = create_sphere(cols, rows, radius=radius)
	vertices = mesh.get_vertices()
	tris = mesh.get_faces()
	cl = np.linspace(-radius, radius, nbr_level+2)[1:-1]
	#scene.visuals.Isoline(vertices=vertices, tris=tris, data=vertices[:, 2],
	#                      levels=cl, color_lev='winter', parent=view.scene)

	# Add a 3D axis to keep us oriented
	scene.visuals.XYZAxis(parent=view.scene)

	view.camera = scene.TurntableCamera()
	view.camera.set_range((-1, 1), (-1, 1), (-1, 1))

	return app, canvas

