"""
app = Renderer()
app.draw_voxels(voxels,red)
app.draw_voxels(voxels,blue)
app.draw_checkerboard()
app.run()
"""
import time
import numpy as np
import vispy.io as io
import matplotlib.pyplot as plt
from numpy import sin, cos, pi
from marching_cubes import march
from segmentation.utils import figure2image
from vispy import app, scene
from vispy.geometry.generation import create_sphere
from vispy.visuals.transforms import MatrixTransform
from vispy import app, gloo, visuals




class Renderer():
	zaxis = -0.25


	def __init__(self):
		# Create a canvas with a 3D viewport
		keys = 'interactive'
		title = 'Isocurve for Triangular Mesh Example'
		self.canvas = scene.SceneCanvas(keys=keys,title=title)
		self.canvas.show()
		self.view = self.canvas.central_widget.add_view()


	def draw_voxels(self, voxels, color, translate, smoothing=15):
		# First grid the data
	    res = np.amax(voxels[1,:] - voxels[0,:])
	    ux = np.unique(voxels[:,0])
	    uy = np.unique(voxels[:,1])
	    uz = np.unique(voxels[:,2])

	    # Expand the model by one step in each direction
	    ux = np.hstack((ux[0] - res, ux, ux[-1] + res))
	    uy = np.hstack((uy[0] - res, uy, uy[-1] + res))
	    uz = np.hstack((uz[0] - res, uz, uz[-1] + res))

	    # Convert to a grid
	    X, Y, Z = np.meshgrid(ux, uy, uz)

	    # Create an empty voxel grid, then fill in the elements in voxels
	    V = np.zeros(X.shape)
	    N = voxels.shape[0]
	    for ii in range(N):
	            ix = ux == voxels[ii,0]
	            iy = uy == voxels[ii,1]
	            iz = uz == voxels[ii,2]
	            V[iy, ix, iz] = 1

	    # Algorithm expects a cube
	    m,n,p = V.shape
	    size = max(m,n,p)
	    volume = np.zeros((size, size, size))
	    volume[:m,:n,:p] = V
	    start = time.time()
	    verts, normals, faces = march(volume, smoothing)
	    print("Marching_cubes took %.3f seconds"%(time.time()-start))

	    print("Rendering mesh")
	    self.draw_mesh(verts,normals,faces,color, translate)


	def draw_mesh(self, vertices, normals, faces, color, translate):
		# Draw the meshgrid
		view = self.view
		vertices = vertices/60
		nv = vertices.size//3
		vcolor = np.ones((nv, 4), dtype=np.float32)
		vcolor[:, 0] = color[0]
		vcolor[:, 1] = color[1]
		vcolor[:, 2] = color[2]
		mesh = scene.visuals.Mesh(vertices=vertices, faces=faces, vertex_colors=vcolor, shading='smooth', parent=view.scene)
		tr = MatrixTransform()
		tr.rotate(-90, (1, 0, 0))
		tr.translate(translate)
		mesh.transform = tr
		mesh.shininess = 0.001

		# Add a 3D axis to keep us oriented
		ax = scene.visuals.XYZAxis(parent=view.scene)
		tr = MatrixTransform()
		tr.translate((0, 0, self.zaxis))
		ax.transform = tr


	def draw_checkerboard(self):
		view = self.view
		view.camera = scene.TurntableCamera(fov=40, elevation=14.0, distance=8)
		view.camera.azimuth = -45

		# Draw the chessboard
		# Build a checkerboard colored square plane with "segments" number of tiles per side.
		size = 0.5
		segments = 8
		white = (0.8, 0.8, 0.98, 0.8)
		black = (0.26, 0.26, 0.39, 0.8)
		materials = [white, black]

		# The transform matrix
		tr = MatrixTransform()
		tr.translate((0, 0, self.zaxis))

		for x in range(-1,segments):
			for y in range(-1,segments):
				i = x * segments + y
				j = 2 * i
				color = materials[(x+y) % 2]
				position = size * np.array([[x,y,0], [x+1,y,0], [x+1,y+1,0], [x,y+1,0]])
				polygon = scene.visuals.Polygon(position, color=color, parent=view.scene)
				polygon.transform = tr

	def run(self):
		app.run()

