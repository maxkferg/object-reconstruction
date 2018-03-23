import time
import numpy as np
import matplotlib.pyplot as plt
import vispy.io as io
from numpy import sin, cos, pi
from skimage import measure
from marching_cubes import march
from .render import render



def axis_equal(ax, X, Y, Z):
    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0

    mid_x = (X.max()+X.min()) * 0.5
    mid_y = (Y.max()+Y.min()) * 0.5
    mid_z = (Z.max()+Z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)


def plot_surface(voxels, voxel_size = 0.1, smoothing=10, is_render=False):
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

    app, canvas = render(verts,normals,faces)
    img = canvas.render()
    img = img[:,:,:3] # Remove alpha

    if is_render:
        io.write_png("example.png",img)
        app.run()

    return img
