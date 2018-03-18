import numpy as np
import space_carving.plotting as plotting
import space_carving.main as carving
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
    xlim = [-50.0107705 , 50.59380309]
    ylim = [-50.907882,   20.91827464]
    zlim = [-20.88477288763805, 32.23735373028762]

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

        print("Showing voxels")
        #plotting.plot_surface(voxels)

        print("Finished carving with",camera.name)

    return plotting.plot_surface(voxels)



def get_carved_image(cameras, silhouettes):
    """Return a image of the carved voxels"""
    return carve(cameras,silhouettes)



