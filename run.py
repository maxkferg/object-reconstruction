import numpy as np
import cv2
import glob

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*6,3), np.float32)
objp[:36, :2] = np.mgrid[0:6,0:6].T.reshape(-1,2)

print("objpoints",objp)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

note_images = sorted(glob.glob('photos/note/cb*.jpg'))
iphone_images = sorted(glob.glob('photos/iphone/cb*.jpg'))
print(note_images)

for fname in iphone_images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (6,6), None)

    # If found, add object points, image points (after refining them)
    if ret is True:
        corners2 = cv2.cornerSubPix(gray, corners,(11,11),(-1,-1),criteria)

        imgpoints.append(corners2)
        objpoints.append(objp)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (6,6), corners2,ret)

        output = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)

        cv2.imshow('img',output)
        cv2.waitKey(0)


cv2.destroyAllWindows()

# Fix up the objpoints
#objpoints[0] = objpoints[0]#*15 # Scale is 15mm
#objpoints[1] = objpoints[1]#*15 # Scale is 15mm
#objpoints[0][:,2] = 300 # The first image is in front
#objpoints[1][:,2] = 400 # The second image is in behind


ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)


def reprojection_error(objpoints, imgpoints, rvecs, tvecs, mtx, dist):
    # Calculate reprojection error
    total_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)

        error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        total_error += error

    mean_error = total_error/len(objpoints)
    print("Reprojection error: %.4f"%mean_error)\

reprojection_error(objpoints, imgpoints, rvecs, tvecs, mtx, dist)



for i,image in enumerate(sorted(glob.glob('photos/iphone/*.jpg'))):
    continue
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







def draw(img, corners, imgpts):
    imgpts = np.int32(imgpts).reshape(-1,2)

    # draw ground floor in green
    img = cv2.drawContours(img, [imgpts[:4]],-1,(0,255,0),-3)

    # draw pillars in blue color
    for i,j in zip(range(4),range(4,8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255),3)

    # draw top layer in red color
    img = cv2.drawContours(img, [imgpts[4:]],-1,(0,0,255),3)

    return img









def reprojection_error2(objpoints, imgpoints, rvecs, tvecs, mtx, dist):
    # Calculate reprojection error
    total_error = 0
    imgpoints2, _ = cv2.projectPoints(objpoints, rvecs, tvecs, mtx, dist)
    error = cv2.norm(imgpoints, imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    print("Reprojection error: %.4f"%error)





objpoints[0] = objpoints[0] # Scale is 15mm
objpoints[1] = objpoints[1] # Scale is 15mm
objpoints[0][:,2] = 0 # The first image is in front
objpoints[1][:,2] = 0 # The second image is in behind


objpoints = np.vstack(objpoints)
imgpoints = np.vstack(imgpoints)


criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((6*6,3), np.float32)
objp[:,:2] = np.mgrid[0:6,0:6].T.reshape(-1,2)


axis = np.float32([[0,0,0], [0,3,0], [3,3,0], [3,0,0],
                   [0,0,-3],[0,3,-3],[3,3,-3],[3,0,-3] ])



# Get the rotation and translation matricsz



for fname in :
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (6,6), None)

    if ret == True:
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)

        # Find the rotation and translation vectors.
        #ret, rvecs, tvecs = cv2.solvePnP(objpoints, imgpoints, mtx, dist, flags=cv2.SOLVEPNP_ITERATIVE)
        ret, rvecs, tvecs = cv2.solvePnP(objp, corners2, mtx, dist, flags=cv2.SOLVEPNP_ITERATIVE)


        print("rvecs",rvecs)
        print("tvecs",tvecs)
        print("mtx",mtx)
        print("dist",dist)

        # Get reprojection error
        print("-----------------------")
        print(imgpoints.shape)
        reprojection_error2(objpoints, imgpoints, rvecs, tvecs, mtx, dist)

        # project 3D points to image plane
        imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)

        img = draw(img, corners2, imgpts)
        img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)

        cv2.imshow('img',img)
        k = cv2.waitKey(0) & 0xff
        if k == 's':
            cv2.imwrite(fname[:6]+'.png', img)


cv2.destroyAllWindows()

