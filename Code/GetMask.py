import numpy as np
import matplotlib.pyplot as plt
import cv2

def GetMask(image):
    ### You can add any number of points by using 
    ### mouse left click. Delete points with mouse
    ### right click and finish adding by mouse
    ### middle click.  More info:
    ### https://matplotlib.org/api/_as_gen/matplotlib.pyplot.ginput.html
    print(image)
    plt.imshow(image)
    plt.axis('image')
    points = plt.ginput(-1, timeout=-1)
    plt.close()

    ### The code below is based on this answer from stackoverflow
    ### https://stackoverflow.com/a/15343106

    mask = np.zeros(image.shape, dtype=np.uint8)
    roi_corners = np.array([points], dtype=np.int32)
    # fill the ROI so it doesn't get wiped out when the mask is applied
    channel_count = image.shape[2]  # i.e. 3 or 4 depending on your image
    ignore_mask_color = (255,)*channel_count
    cv2.fillPoly(mask, roi_corners, ignore_mask_color)

    return mask


if __name__ == '__main__':
    # Setting up the input output paths
    inputDir = '../Images/'
    outputDir = '../Results/'

    inputImg = '../Images/bung.jpg'
    test = cv2.imread('../Images/bungTest2.jpg')
    # main area to specify files and display blended image

    # index = 1
    # Read data and clean mask
    # source, maskOriginal, target = Read(str(index).zfill(2), inputDir)

    # Cleaning up the mask
    # mask = np.ones_like(maskOriginal)
    # mask[maskOriginal < 0.5] = 0

    currMask = GetMask(test)
    # Writing the result
    # print(np.amax(pyramidOutput))
    # print(np.amin(pyramidOutput))
    # pyramidOutput = np.clip(pyramidOutput, 0, 1)
    plt.imsave(outputDir + "masktest2" + '.jpg', currMask)
    # plt.imsave("maskTest.jpg".format(outputDir, str(index).zfill(2)), currMask)