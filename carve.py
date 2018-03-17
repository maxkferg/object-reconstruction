import numpy as np
import scipy.io as sio
import argparse
import glob
import calibrate
import cv2
from skimage import io
from skimage import color
from space_carving.camera import Camera
from space_carving.plotting import *


# A very simple, but useful method to take the difference between the
# first and second element (usually for 2D vectors)
def diff(x):
    return x[1] - x[0]

"""
TO_HOMOGENEOUS
Convert an array of points to homogeneous coordinates.

Arguments:
    points: An Nx3 array of points

Returns:
    hpoints: An Nx4 array of points
"""
def to_homogeneous(points):
    if points.ndim != 2:
        raise ValueError("to_homogeneous only accepts 2d arrays")
    if points.shape[1] != 3:
        raise ValueError("The second dimension was not 3")
    ones = np.ones((points.shape[0],1))
    hpoints = np.hstack((points,ones))
    return hpoints





'''
FORM_INITIAL_VOXELS  create a basic grid of voxels ready for carving

Arguments:
    xlim - The limits of the x dimension given as [xmin xmax]

    ylim - The limits of the y dimension given as [ymin ymax]

    zlim - The limits of the z dimension given as [zmin zmax]

    num_voxels - The approximate number of voxels we desire in our grid

Returns:
    voxels - An ndarray of size (N, 3) where N is approximately equal the
        num_voxels of voxel locations.

    voxel_size - The distance between the locations of adjacent voxels
        (a voxel is a cube)

Our initial voxels will create a rectangular prism defined by the x,y,z
limits. Each voxel will be a cube, so you'll have to compute the
approximate side-length (voxel_size) of these cubes, as well as how many
cubes you need to place in each dimension to get around the desired
number of voxel. This can be accomplished by first finding the total volume of
the voxel grid and dividing by the number of desired voxels. This will give an
approximate volume for each cubic voxel, which you can then use to find the
side-length. The final "voxels" output should be a ndarray where every row is
the location of a voxel in 3D space.
'''
def form_initial_voxels(xlim, ylim, zlim, num_voxels):
    # TODO: Implement this method!
    # TODO: Use the center of each voxel
    cube_volume = np.prod([d[1]-d[0] for d in (xlim,ylim,zlim)])
    voxel_volume = cube_volume / num_voxels
    voxel_size = voxel_volume**(1./3)
    x_locations = np.arange(xlim[0], xlim[1], voxel_size)
    y_locations = np.arange(ylim[0], ylim[1], voxel_size)
    z_locations = np.arange(zlim[0], zlim[1], voxel_size)
    voxels = np.array(np.meshgrid(x_locations, y_locations, z_locations)).T.reshape(-1,3)
    return voxels, voxel_size
'''
GET_VOXEL_BOUNDS: Gives a nice bounding box in which the object will be carved
from. We feed these x/y/z limits into the construction of the inital voxel
cuboid.

Arguments:
    cameras - The given data, which stores all the information
        associated with each camera (P, image, silhouettes, etc.)

    estimate_better_bounds - a flag that simply tells us whether to set tighter
        bounds. We can carve based on the silhouette we use.

    num_voxels - If estimating a better bound, the number of voxels needed for
        a quick carving.

Returns:
    xlim - The limits of the x dimension given as [xmin xmax]

    ylim - The limits of the y dimension given as [ymin ymax]

    zlim - The limits of the z dimension given as [zmin zmax]

The current method is to simply use the camera locations as the bounds. In the
section underneath the TODO, please implement a method to find tigther bounds:
One such approach would be to do a quick carving of the object on a grid with
very few voxels. From this coarse carving, we can determine tighter bounds. Of
course, these bounds may be too strict, so we should have a buffer of one
voxel_size around the carved object.
'''
def get_voxel_bounds(cameras, estimate_better_bounds = False, num_voxels = 4000):
    camera_positions = np.vstack([c.T for c in cameras])
    xlim = [camera_positions[:,0].min(), camera_positions[:,0].max()]
    ylim = [camera_positions[:,1].min(), camera_positions[:,1].max()]
    zlim = [camera_positions[:,2].min(), camera_positions[:,2].max()]

    # For the zlim we need to see where each camera is looking.
    camera_range = 0.6 * np.sqrt(diff( xlim )**2 + diff( ylim )**2)
    for c in cameras:
        viewpoint = c.T - camera_range * c.get_camera_direction()
        print('viewpoint',viewpoint)
        zlim[0] = min( zlim[0], viewpoint[2] )
        zlim[1] = max( zlim[1], viewpoint[2] )

    # Move the limits in a bit since the object must be inside the circle
    xlim = xlim + diff(xlim) / 4 * np.array([1, -1])
    ylim = ylim + diff(ylim) / 4 * np.array([1, -1])

    if estimate_better_bounds:
        # TODO: Implement this method!
        raise Exception('Not Implemented Error')
    return xlim, ylim, zlim


'''
CARVE: carves away voxels that are not inside the silhouette contained in
    the view of the camera. The resulting voxel array is returned.

Arguments:
    voxels - an Nx3 matrix where each row is the location of a cubic voxel

    camera - The camera we are using to carve the voxels with. Useful data
        stored in here are the "silhouette" matrix, "image", and the
        projection matrix "P".

Returns:
    voxels - a subset of the argument passed that are inside the silhouette
'''
def carve(voxels, camera):
    # TODO: Implement this method!
    N = voxels.shape[0]
    voxel_index = np.arange(0, N)
    points2d = camera.project(voxels)
    points2d = points2d.astype(int)

    print("Points 2D")
    print(points2d)

    # Discard any voxels outside the bounds
    h,w = camera.silhouette.shape

    # Split into x and y coordinates
    voxel_x = points2d[:,0]
    voxel_y = points2d[:,1]

    # Keep items where both columns are valid indices
    filter1 = np.logical_and(voxel_x >= 0, voxel_x < w)
    filter2 = np.logical_and(voxel_y >= 0, voxel_y < h)
    valid_indices = np.logical_and(filter1,filter2)

    # Keep the valid points and the corrosponding voxel indices
    points2d = points2d[valid_indices,:]
    voxel_index = voxel_index[valid_indices]

    # Find the voxels which are filled (not background)
    xi = points2d[:,1]
    yi = points2d[:,0]
    silhouette_filter = (camera.silhouette[xi, yi] == 1)

    # Find the indices of those voxels
    voxel_index = voxel_index[silhouette_filter]

    return voxels[voxel_index, :]



'''
ESTIMATE_SILHOUETTE: Uses a very naive and color-specific heuristic to generate
the silhouette of an object

Arguments:
    im - The image containing a known object. An ndarray of size (H, W, C).

Returns:
    silhouette - An ndarray of size (H, W), where each pixel location is 0 or 1.
        If the (i,j) value is 0, then that pixel location in the original image
        does not correspond to the object. If the (i,j) value is 1, then that
        that pixel location in the original image does correspond to the object.
'''
def estimate_silhouette(im):
    return np.logical_and(im[:,:,0] > im[:,:,2], im[:,:,0] > im[:,:,1] )


def carve_dragon():
    """Carve the dragon shape"""
    estimate_better_bounds = False
    use_true_silhouette = True
    frames = sio.loadmat('space_carving/frames.mat')['frames'][0]
    cameras = [Camera(x) for x in frames]

    # Generate the silhouettes based on a color heuristic
    if not use_true_silhouette:
        for i, c in enumerate(cameras):
            c.true_silhouette = c.silhouette
            c.silhouette = estimate_silhouette(c.image)
            if i == 0:
                plt.figure()
                plt.subplot(121)
                plt.imshow(c.true_silhouette, cmap = 'gray')
                plt.title('True Silhouette')
                plt.subplot(122)
                plt.imshow(c.silhouette, cmap = 'gray')
                plt.title('Estimated Silhouette')
                plt.show()

    # Generate the voxel grid
    # You can reduce the number of voxels for faster debugging, but
    # make sure you use the full amount for your final solution
    #num_voxels = 6e6
    num_voxels = 4e6
    xlim, ylim, zlim = get_voxel_bounds(cameras, estimate_better_bounds)
    print(xlim, ylim, zlim)

    # This part is simply to test forming the initial voxel grid
    voxels, voxel_size = form_initial_voxels(xlim, ylim, zlim, 4000)
    plot_surface(voxels)
    voxels, voxel_size = form_initial_voxels(xlim, ylim, zlim, num_voxels)

    # Test the initial carving
    voxels = carve(voxels, cameras[0])
    #if use_true_silhouette:
    #    plot_surface(voxels)

    # Result after all carvings
    for c in cameras:
        voxels = carve(voxels, c)
    plot_surface(voxels, voxel_size)



def carve_stapler():
    """Carve the dragon shape"""
    estimate_better_bounds = False

    note_images = sorted(glob.glob('photos/note/cb*.jpg'))
    note_test = 'photos/note/all.jpg'

    iphone_images = sorted(glob.glob('photos/iphone/cb*.jpg'))
    iphone_test = 'photos/iphone/all.jpg'

    cameras = [
        calibrate.get_camera(note_images, note_test, debug=False),
        calibrate.get_camera(iphone_images, iphone_test, debug=False)
    ]

    cameras[0].image = plt.imread('photos/note/stapler-mask.jpg')
    cameras[1].image = plt.imread('photos/iphone/stapler-mask.jpg')

    cameras[0].silhouette = color.rgb2gray(plt.imread('photos/note/stapler-mask.jpg'))
    cameras[1].silhouette = color.rgb2gray(plt.imread('photos/iphone/stapler-mask.jpg'))

    cameras[0].T = np.squeeze(cameras[0].T.T)
    cameras[1].T = np.squeeze(cameras[1].T.T)

    cameras[0].T = np.array([0,0,0],dtype=float)
    #cameras[1].T = np.array([0,0,0],dtype=float)

    # Show silhouettes
    plt.figure()
    for i, c in enumerate(cameras):
        plt.subplot(121+i)
        print('-->',c.silhouette)
        plt.imshow(c.silhouette, cmap = 'gray')
        plt.title('Silhouette')
    plt.show()

    # Generate the voxel grid
    # You can reduce the number of voxels for faster debugging, but
    # make sure you use the full amount for your final solution
    #num_voxels = 6e6
    num_voxels = 6e5
    xlim, ylim, zlim = get_voxel_bounds(cameras, estimate_better_bounds)
    print("bounds:", xlim, ylim, zlim)
    xlim = [-6,6]
    ylim = [-6,6]
    zlim = [-6,6]

    # This part is simply to test forming the initial voxel grid
    #voxels, voxel_size = form_initial_voxels(xlim, ylim, zlim, 4000)
    #plot_surface(voxels)
    voxels, voxel_size = form_initial_voxels(xlim, ylim, zlim, num_voxels)

    # Test the initial carving
    voxels = carve(voxels, cameras[0])
    plot_surface(voxels)

    voxels = carve(voxels, cameras[1])
    plot_surface(voxels)


if __name__ == '__main__':
    carve_dragon()
    #carve_stapler()