import time
import numpy as np
import space_carving.plotting as plotting
import space_carving.simple as carving
import matplotlib.pyplot as plt
from segmentation.utils import figure2image


def show_silhouette(camera, silhouette):
    """Show silhouettes"""
    plt.figure()
    plt.imshow(silhouette, cmap = 'gray')
    plt.title('Silhouette from ' + camera.name)
    plt.show()



def carve(cameras,silhouettes):
    """
    Carve an object from voxels
    Return a figure object (not shown)
    """
    debug = True
    estimate_better_bounds = False

    # Generate the voxel grid
    # You can reduce the number of voxels for faster debugging, but
    # make sure you use the full amount for your final solution
    #num_voxels = 6e6
    num_voxels = 6e6
    xlim, ylim, zlim = carving.get_voxel_bounds(cameras, estimate_better_bounds)
    print("bounds:", xlim, ylim, zlim)
    xlim = [-100.0107705 , 100.59380309]
    ylim = [-100.907882,   100.91827464]
    zlim = [-30.884772,   30.2373537302]

    # This part is simply to test forming the initial voxel grid
    voxels, voxel_size = carving.form_initial_voxels(xlim, ylim, zlim, num_voxels)

    # Test the initial carving
    for i in range(len(cameras)):
        camera = cameras[i]
        silhouette = silhouettes[i]

        # Show silhoute
        # show_silhouette(camera, silhouette)

        print("Carving voxels")
        voxels = carving.carve(voxels, camera, silhouette)
        print("Finished carving with",camera.name)
    return voxels



def get_carved_image(cameras, silhouettes, verbose=True, debug=True):
    """Return a image of the carved voxels"""
    start = time.time()
    voxels = carve(cameras,silhouettes)
    print("Voxel carving took %.3f"%(time.time()-start))

    start = time.time()
    image = plotting.plot_surface(voxels, debug=debug)
    print("Voxel rendering took %.3f"%(time.time()-start))

    return image




