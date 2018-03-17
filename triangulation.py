import numpy as np
import cv2
import glob
import calibrate
import argparse
import cv2


def calibrate():
    note_images = sorted(glob.glob('photos/note/cb*.jpg'))
    note_test = 'photos/note/all.jpg'

    iphone_images = sorted(glob.glob('photos/iphone/cb*.jpg'))
    iphone_test = 'photos/iphone/all.jpg'

    print("Calibrating Note")
    calibrate.calibrate(note_images, note_test)

    print("Calibrating iPhone")
    calibrate.calibrate(iphone_images, iphone_test)


# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not
refPt = []
cropping = False


def click_and_crop(event, x, y, flags, param):
    # grab references to the global variables
    global refPt, cropping
    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
    if event == cv2.EVENT_LBUTTONDOWN:
        print("Started at:",x,y)
        refPt = [(x, y)]
        cropping = True
    # check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates
        print("Ended at:",x,y)
        refPt.append((x, y))


def manually_get_bounding_box(impath,title):
    """Get the bounding boxes for the iPhone and note images"""
    # load the image, clone it, and setup the mouse callback function
    image = cv2.imread(impath)
    image = cv2.resize(image, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_CUBIC)
    clone = image.copy()
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", click_and_crop)

    print("---------------------")
    print(title)
    print("---------------------")
    print("Click and drag to select a region")
    print("Press c when done or r to reset")

    # keep looping until the 'q' key is pressed
    while True:
        # display the image and wait for a keypress
        cv2.imshow("image", image)
        key = cv2.waitKey(1) & 0xFF

        # if the 'r' key is pressed, reset the cropping region
        if key == ord("r"):
            image = clone.copy()

        # if the 'c' key is pressed, break from the loop
        elif key == ord("c"):
            if len(refPt)<2:
                print("Not enough points")
            else:
                break
    return refPt



def get_all_boxes():
    """Return bounding boxes in form [x1,y1,x2,y2]"""
    iphone = {
        "stapler": [52,397,587,719],
        "mouse": [609,492,1034,704],
        "ball": [1114,344,1559,727],
    }
    note = {
        "stapler": [851,265,1178,639],
        "mouse": [1250,377,1516,577],
        "ball": [1571,257,1855,536],
    }
    return iphone,note



def draw_bounding_boxes(impath, boxes, title="Bounding boxes"):
    img = cv2.imread(impath)
    img = cv2.resize(img, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_CUBIC)
    colors = [(255,0,0),(0,255,0),(0,0,255)]
    for i,name in enumerate(boxes):
        box = boxes[name]
        pt1 = tuple(box[0:2])
        pt2 = tuple(box[2:4])
        cv2.rectangle(img,pt1,pt2,colors[i],3)
        cv2.imshow('title', img)
    key = cv2.waitKey(0)
    cv2.destroyAllWindows()




if __name__=="__main__":
    iphone_test = 'photos/iphone/all.jpg'
    note_test = 'photos/note/all.jpg'
    #manually_get_bounding_box(iphone_test,"iphone stapler")
    #manually_get_bounding_box(iphone_test,"iphone ball")
    #manually_get_bounding_box(iphone_test,"iphone mouse")

    #manually_get_bounding_box(note_test,"note stapler")
    #manually_get_bounding_box(note_test,"note ball")
    #manually_get_bounding_box(note_test,"note mouse")

    iphone_boxes, note_boxes = get_all_boxes()
    draw_bounding_boxes(iphone_test, iphone_boxes, "iPhone")
    draw_bounding_boxes(note_test, note_boxes, "note")

