

# Import required libraries
import numpy as np
import cv2
from skimage.transform import resize
from matplotlib import pyplot as plt
# np.set_printoptions(threshold=np.inf)
# Read source, target and mask for a given id
def Read(id, path = ""):
    # source = plt.imread(path + "source_" + id + ".jpg")
    # info = np.iinfo(source.dtype) # get information about the image type (min max values)
    # source = source.astype(np.float32) / info.max # normalize the image into range 0 and 1
    # target = plt.imread(path + "target_" + id + ".jpg")
    # info = np.iinfo(target.dtype) # get information about the image type (min max values)
    # target = target.astype(np.float32) / info.max # normalize the image into range 0 and 1
    # mask   = plt.imread(path + "mask_" + id + ".jpg")
    # info = np.iinfo(mask.dtype) # get information about the image type (min max values)
    # mask = mask.astype(np.float32) / info.max # normalize the image into range 0 and 1
    
    source = plt.imread(path + "corbTest2.jpg")
    info = np.iinfo(source.dtype) # get information about the image type (min max values)
    source = source.astype(np.float32) / info.max # normalize the image into range 0 and 1
    target = plt.imread(path + "spenTest3.jpg")
    info = np.iinfo(target.dtype) # get information about the image type (min max values)
    target = target.astype(np.float32) / info.max # normalize the image into range 0 and 1
    mask   = plt.imread(path + "maskSC2.jpg")
    info = np.iinfo(mask.dtype) # get information about the image type (min max values)
    mask = mask.astype(np.float32) / info.max # normalize the image into range 0 and 1

    return source, mask, target
# Question!!!
# How to properly pad? - use np.pad???
# Does this stuff look properly implemented / right idea?
# How to collapse pyramid, slides confuse me with the diff pics adding together?
# What are "failures" again? - what do the operatons do? 

# Pyramid Blend
def PyramidBlend(source, mask, target, levels):
    # currHeight, currWidth, currNoth = mask.shape
    # nextHeight, nextWidth, nextNoth = mask.shape
    # while(True):
    #     if (nextHeight % 9 == 0):
    #         break
    #     else:
    #         nextHeight += 1
    # newShape = (nextHeight, nextHeight, nextNoth)
    # b = np.ones((nextHeight,nextHeight,3))
    # maskPad = pad(mask.shape, b)
    # maskPad = np.pad(nextHeight, 6, mode='constant')

    gausMask = gaussianPyr(mask, levels)
    gausTarget = gaussianPyr(target, levels)
    gausSource = gaussianPyr(source, levels)
    # print(gausTarget)

    lapTarget = lapPyr(gausTarget, levels)
    lapSource = lapPyr(gausSource, levels)
    LCArr = []

    # for each level compute 
    for i in range(levels):
        LC = gausMask[i] * lapSource[i] + (1 - gausMask[i]) * lapTarget[i]

        # Add to array to build pyramid to collapse 
        LCArr.append(LC)

    final = collapsePyr(LCArr, levels)
    return final

# create gaussian pyramid using mask w/ pyr down (4 levels)
def gaussianPyr(img, levels):
    imgArray = [img]
  
    # img1, img2, img3 = img.shape
    # levels - 1 since OG img is already in array, downsample twice 
    for i in range(levels - 1):
        # pyr method
        # img = cv2.pyrDown(img)

        # resize method
        # Downsample OG image 
        nextHeight, nextWidth, nextNoth = img.shape
        newShape = (np.round(nextHeight / 2), np.round(nextWidth / 2))
        img = resize(img, newShape)

        # print("img")
        # print(img.shape)
        # Add to array to build pyramid, repeat for X levels
        imgArray.append(img)
    return imgArray


def lapPyr(imgArray, levels):
    # print(imgArray[3])
    # [0, 1, 2 ,3] - 0 is OG image | 3 is smallest
    # currImg = imgArray[0]

    outputArray = []
    for i in range(levels):
        # smallest img L4 = G4 right?
        if (i == (levels - 1)):
            outputArray.append(imgArray[(levels - 1)])
            break

        # following slides 21 of 05_Pyr

        # pyr method
        # currImg = imgArray[i]
        # nextImg = imgArray[i + 1]
        # nextImg = cv2.pyrUp(nextImg)

        # resize method
        # print("nextImg2")
        # print(nextImg.shape)

        # Get the biggest version of image
        currImg = imgArray[i]

        # Resize the next smaller version of the image to current bigger size
        currHeight, currWidth, currNoth = currImg.shape
        nextShape = (currHeight, currWidth) #(np.round(nextHeight * 2), np.round(nextWidth * 2))
        nextImg = resize(imgArray[i + 1], nextShape)

        # Take the difference and add to pyramid array, repeat
        outputImg = currImg - nextImg
        outputArray.append(outputImg)

    return outputArray

def collapsePyr(imgArray, levels):
    counter = levels - 1
    print('imgArray')
    print(imgArray[2].shape)
    outputArray = []
    outputImg = imgArray[-1]
    # following slides 24 
    for i in range(levels - 2, -1, -1):
        # if (counter == 0):
        #     break
        
        #pyr method
        # currImg = cv2.pyrUp(imgArray[counter])
        # nextImg = imgArray[counter - 1]
       
        #resize method
        # currHeight, currWidth = imgArray[counter].shape

        # Get smallest version of image
        # currImg = imgArray[i]
        # Get the next image in the array (idk if right) 
        nextImg = imgArray[i]
        
        # print('Small Ver')
        # print(currImg.shape)

        # Upscale Small Version to the Next larger image size
        nextHeight, nextWidth, nextNoth = nextImg.shape
        nextShape = (nextHeight, nextWidth)
        outputImg = resize(outputImg, nextShape)

        # print('Upscale ver')
        # print(currImg.shape)
        # print('Next Image')
        # print(nextImg.shape)
        # currImg += currImg[i]
        #Add together, repeat
        outputImg += nextImg
        # print('test')
        # print(outputImg.shape)


        # counter -= 1

    return outputImg

if __name__ == '__main__':
    # Setting up the input output paths
    inputDir = '../Images/'
    outputDir = '../Results/'

    # main area to specify files and display blended image

    index = 1
    levels = 3
    # Read data and clean mask
    # maskOriginal = Read(str(index).zfill(2), inputDir)
    source, maskOriginal, target = Read(str(index).zfill(2), inputDir)

    # target = cv2.imread('../Images/bungTest2.jpg')
    # maskOriginal = cv2.imread('../Images/maskTest3.jpg')
    # source = cv2.imread('../Images/bungTest2.jpg')
    # Cleaning up the mask
    mask = np.ones_like(maskOriginal)
    mask[maskOriginal < 0.5] = 0

    ### The main part of the code ###
    # Implement the PyramidBlend function (Task 2)
    pyramidOutput = PyramidBlend(source, mask, target, levels)
    # pyramidOutput = collapsePyr(pyramidOutput)

    
    # Writing the result
    # print(np.amax(pyramidOutput))
    # print(np.amin(pyramidOutput))
    pyramidOutput = np.clip(pyramidOutput, 0, 1)
    plt.imsave("naiveBlend.jpg".format(outputDir, str(index).zfill(2)), pyramidOutput)
#use cv2.pyr down for gaussian 

# for lap build gaus pyramid, then starting from the bottom (lower res), go up, up sample 
# and find the differnece between upsample and take the diffenec between the level above and upsample.
# Then difference for s and t compute gaus for mask, mask will have diff res, for every res combine the 2, use the mask at the specific res
# Then the image will be the sum of all the differences of everything 

# pyr up = upsample

# Compute Laplacian pyramids LS and LT from the source and target images. 
# peer up and peer down
# lap - compute gaus and upscale from previous level ?
#   use skimage.transform.resize to form x levels?
# Compute a Gaussian pyramid GM from the mask.
# use gaus to mix 2 lap 
# 
# Use the the Gaussian pyramid to combine the source and target Laplacian pyramids as follows:
#   LC(l) = GM(l) × LS(l) + (1 − GM(l)) × LT (l), where l is the layer index.

# Collapse the blended pyramid LC to reconsruct the final blended image