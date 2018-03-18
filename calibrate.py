import os
import cv2
import glob
import time
import numpy as np
from space_carving.camera import Camera

CALIBRATION_SIZE = (8,6)


def multiview_reprojection_error(objpoints, imgpoints, rvecs, tvecs, mtx, dist):
    # Calculate reprojection error
    total_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)

        error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        total_error += error

    mean_error = total_error/len(objpoints)
    return mean_error



def reprojection_error(objpoints, imgpoints, rvec, tvec, mtx, dist):
    """Calculate reprojection error with a single view"""
    total_error = 0
    imgpoints2, _ = cv2.projectPoints(objpoints, rvec, tvec, mtx, dist)
    error = cv2.norm(imgpoints, imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    return error



def draw_cube(img, corners, imgpts):
    imgpts = np.int32(imgpts).reshape(-1,2)

    # draw ground floor in green
    img = cv2.drawContours(img, [imgpts[:4]],-1,(0,255,0),-3)

    # draw pillars in blue color
    for i,j in zip(range(4),range(4,8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255),3)

    # draw top layer in red color
    img = cv2.drawContours(img, [imgpts[4:]],-1,(0,0,255),3)

    return img



def draw_axis(img, imgpts):
    """Draw an axis located at imgponts[0]"""
    origin = tuple(imgpts[0].ravel())
    img = cv2.line(img, origin, tuple(imgpts[1].ravel()), (255,0,0), 5)
    img = cv2.line(img, origin, tuple(imgpts[2].ravel()), (0,255,0), 5)
    img = cv2.line(img, origin, tuple(imgpts[3].ravel()), (0,0,255), 5)
    return img



def imshow(img, scale=0.4):
    """Resize and display image
    Wait for enter to continue
    """
    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    cv2.imshow('img',img)
    k = cv2.waitKey(0) & 0xff
    if k == 's':
        cv2.imwrite(fname[:6]+'.png', img)



def get_objp(calibration_size, z=0):
    """Return object points for chessboard"""
    n = calibration_size[0]
    m = calibration_size[1]
    objp = np.zeros((m*n,3), np.float32)
    objp[:,:2] = np.mgrid[0:n,0:m].T.reshape(-1,2)
    objp[:,2] = z
    return objp



def collect_calibration_images(camera, folder, n=100):
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = get_objp(CALIBRATION_SIZE)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    count = 0

    raw_image_folder = os.path.join(folder, "raw")
    test_image_folder = os.path.join(folder, "pattern")

    os.makedirs(raw_image_folder)
    os.makedirs(test_image_folder)

    for img in range(n):

        # Get a camera image
        img = camera.get_image(verbose=True)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, CALIBRATION_SIZE, None)

        # If found, add object points, image points (after refining them)
        if ret is not True:
            print("Failed to find chessboard")
            time.sleep(1)
        else:
            # Save original image
            filename = os.path.join(folder, "raw", "%i.png"%count)
            cv2.imwrite(filename, img)

            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (7,9), corners2,ret)

            # Draw the calibration image
            filename = os.path.join(folder, "pattern", "%i.png"%count)
            cv2.imwrite(filename, img)
            count+=1



def calibrate_camera_intrinsics(camera, folder, debug=False):
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = get_objp(CALIBRATION_SIZE)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    image_folder = os.path.join(folder, "raw")

    for filename in os.listdir(image_folder):

        img_path = os.path.join(image_folder, filename)
        img = cv2.imread(img_path,-1)

        if img is None:
            raise ValueError("Can not read file", img_path)

        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        print("Searching for chessboard corners")
        ret, corners = cv2.findChessboardCorners(gray, CALIBRATION_SIZE, None)

        # If found, add object points, image points (after refining them)
        if ret is True:
            corners2 = cv2.cornerSubPix(gray, corners,(11,11),(-1,-1),criteria)
            imgpoints.append(corners2)
            objpoints.append(objp)

            # Draw and display the corners
            if debug:
                img = cv2.drawChessboardCorners(img, (7,9), corners2,ret)
                imshow('img',img)
                cv2.destroyAllWindows()
        else:
            print("Could not find chessboard corners")

    cv2.destroyAllWindows()

    # Perform the calibration
    print("Performing calibration")
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    if ret is None:
        raise ValueError("Failed to calibrate ", camera.name)

    # Store the intrisic paramereters
    camera.set_intrinsics(mtx, dist)
    print("Successfully calibrated ",camera.name)

    # Return the calibrated camera
    return camera



def take_extrinsic_photo(camera, path, debug=True):
    """
    Take a photo for extrisic calculation
    Check that the chessboard can be found
    Save it to the correct folder
    """
    # Arrays to store object points and image points from all the images.
    imgpoints = [] # 2d points in image plane.
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    message = "Press enter to take photo = {0}"
    input(message.format(path))

    while True:
        # Take an image of the extrinsic
        img = camera.get_image()

        if img is None:
            raise ValueError("Can not read image from camera", img)

        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        print("Searching for chessboard corners")
        ret, corners = cv2.findChessboardCorners(gray, CALIBRATION_SIZE, None)

        # If found, add object points, image points (after refining them)
        if ret is True:
            corners2 = cv2.cornerSubPix(gray, corners,(11,11),(-1,-1),criteria)
            chess_image = img.copy()
            chess_image = cv2.drawChessboardCorners(chess_image, CALIBRATION_SIZE, corners2, ret)
            # Draw and display the corners
            if debug:
                imshow(chess_image)
                cv2.destroyAllWindows()
            # Break as soon as we have a good photo
            break
        else:
            print("Could not find chessboard corners")

    # Save the chessboard corners
    print("Saving file to ", path)
    cv2.imwrite(path, img)

    # Save the chess image as well
    chess_path = path.replace(".png","-chess.png")
    print("Saving chess version", chess_path)
    cv2.imwrite(chess_path, chess_image)

    # Return the calibrated camera
    return camera




def calibrate_camera_extrinsics(camera, config, debug=True):
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    for item in [config[0]]:
        z = item["z"]
        img_path = item["file"]

        # prepare object points
        objp = get_objp(CALIBRATION_SIZE, z=z)

        img = cv2.imread(img_path,-1)
        #img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)

        if img is None:
            raise ValueError("Can not read file", img_path)

        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        print("Searching for chessboard corners")
        ret, corners = cv2.findChessboardCorners(gray, CALIBRATION_SIZE, None)

        # If found, add object points, image points (after refining them)
        if ret is True:
            corners2 = cv2.cornerSubPix(gray, corners,(11,11),(-1,-1),criteria)
            imgpoints.append(corners2)
            objpoints.append(objp)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, CALIBRATION_SIZE, corners2,ret)

            if debug:
                imshow(img)
                cv2.destroyAllWindows()
        else:
            print("Could not find chessboard corners")

    objpoints = np.vstack(objpoints)
    imgpoints = np.vstack(imgpoints)

    print("\nCalibrating extrinsics ... ")
    ret, rvec, tvec = cv2.solvePnP(objpoints, imgpoints, camera.mtx, camera.dist, flags=cv2.SOLVEPNP_ITERATIVE)

    if ret is not None:
        camera.set_extrinsics(rvec, tvec)
        draw_cube_on_chessboard(camera, img_path)
        print("Successfully calibrated extrinsics for:", camera.name)
    else:
        raise ValueError("Failed to calibrate ", camera.name)

    # Return the calibrated camera
    return camera




def calibrate(chessboard_image, test_image, debug=True):
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = get_objp(CALIBRATION_SIZE)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    img = chessboard_image
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    print("Searching for chessboard corners")
    ret, corners = cv2.findChessboardCorners(gray, CALIBRATION_SIZE, None)
    print("Done")

    # If found, add object points, image points (after refining them)
    if ret is True:
        corners2 = cv2.cornerSubPix(gray, corners,(11,11),(-1,-1),criteria)

        imgpoints.append(corners2)
        objpoints.append(objp)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (7,9), corners2,ret)

        output = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)

        if debug:
            cv2.imshow('img',output)
            cv2.waitKey(0)

    else:
        raise ValueError("Could not find chessboard corners")

    cv2.destroyAllWindows()

    # Perform the calibration
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    # Print the calibration error
    error = multiview_reprojection_error(objpoints, imgpoints, camera.r, camera.t, mtx, dist)
    print("Calibration reprojection error: %.4f"%error)

    if debug:
        print("Stacking calibration points to determine rotation")
    objpoints = np.vstack(objpoints)
    imgpoints = np.vstack(imgpoints)


    # Get the rotation and translation matricsz
    print("\nCalibrating...")
    ret, rvec, tvec = cv2.solvePnP(objpoints, imgpoints, mtx, dist, flags=cv2.SOLVEPNP_ITERATIVE)

    # Stats
    print("Success: ",ret)
    print("Rvec:\n",rvec)
    print("Tvec:\n",tvec)
    print("Reprojection Error:",reprojection_error(objpoints, imgpoints, rvec, tvec, mtx, dist))

    # Draw cube
    if debug:
        box = np.float32([[0,0,0], [0,3,0], [3,3,0], [3,0,0],
                           [0,0,-3],[0,3,-3],[3,3,-3],[3,0,-3] ])

        imgpts, _ = cv2.projectPoints(box, rvec, tvec, mtx, dist)
        img = test_image
        img = draw_cube(img, None, imgpts)
        imshow(img)
        cv2.destroyAllWindows()

    return rvec, tvec, mtx, dist



def draw_axis_on_image(camera, image_file, out_file):
    """
    Draw a cube on the chessboard for testing purposes
    Returns
    """
    img = cv2.imread(image_file)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Define the box coordinates
    axis = np.float32([[0,0,0], [6,0,0], [0,6,0], [0,0,-6]]).reshape(-1,3)

    # project 3D points to image plane
    imgpts, _ = cv2.projectPoints(axis, camera.r, camera.t, camera.mtx, camera.dist)
    rpe = reprojection_error(axis, imgpts, camera.r, camera.t, camera.mtx, camera.dist)
    print("Axis reprojection error:",rpe)

    img = draw_axis(img, imgpts)

    cv2.imwrite(out_file, img)

    return img


def draw_cube_on_chessboard(camera, filename, debug=True):
    """
    Draw a cube on the chessboard for testing purposes
    Returns
    """
    img = cv2.imread(filename)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, CALIBRATION_SIZE, None)

    # Define the box coordinates
    axis = np.float32([[0,0,0], [0,3,0], [3,3,0], [3,0,0],
                       [0,0,-3],[0,3,-3],[3,3,-3],[3,0,-3] ])

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = get_objp(CALIBRATION_SIZE)

    if ret is not True:
        raise ValueError("Could not ")

    corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)

    # Find the rotation and translation vectors.
    ret, rvecs, tvecs = cv2.solvePnP(objp, corners2, camera.mtx, camera.dist, flags=cv2.SOLVEPNP_ITERATIVE)

    # Get reprojection error
    imgpts, _ = cv2.projectPoints(objp, rvecs, tvecs, camera.mtx, camera.dist)
    rpe = reprojection_error(objp, imgpts, rvecs, tvecs, camera.mtx, camera.dist)
    print("Chessboard reprojection error:",rpe)

    # project 3D points to image plane
    imgpts, _ = cv2.projectPoints(axis, rvecs, tvecs, camera.mtx, camera.dist)
    rpe = reprojection_error(axis, imgpts, rvecs, tvecs, camera.mtx, camera.dist)
    print("Box reprojection error:",rpe)

    img = draw_cube(img, corners2, imgpts)

    if debug:
        imshow(img)

    return rvecs, tvecs




def undistort(images):
    #for i,image in enumerate(sorted(glob.glob('photos/iphone/*.jpg'))):
    for i,image in enumerate(images):
        print("Processing image",image)
        img = cv2.imread(image)
        h,  w = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

        # Undistort the image
        dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

        # crop the image
        x,y,w,h = roi
        dst = dst[y:y+h, x:x+w]
        cv2.imwrite('output/image%i.png'%i,dst)




if __name__=="__main__":
    note_images = sorted(glob.glob('photos/note/cb*.jpg'))
    note_test = 'photos/note/all.jpg'

    iphone_images = sorted(glob.glob('photos/iphone/cb*.jpg'))
    iphone_test = 'photos/iphone/all.jpg'

    print("Calibrating Note")
    calibrate(note_images, note_test)

    print("Calibrating iPhone")
    calibrate(iphone_images, iphone_test)


