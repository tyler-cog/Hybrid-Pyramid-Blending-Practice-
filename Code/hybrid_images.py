"""
Credit: Alyosha Efros
""" 


import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy
import skimage.transform as sktr

isGray = True

def get_points(im1, im2):
    print('Please select 2 points in each image for alignment.')
    plt.imshow(im1)
    p1, p2 = plt.ginput(2)
    plt.close()
    plt.imshow(im2)
    p3, p4 = plt.ginput(2)
    plt.close()
    return (p1, p2, p3, p4)

def recenter(im, r, c):
    R, C, _ = im.shape
    rpad = (int) (np.abs(2*r+1 - R))
    cpad = (int) (np.abs(2*c+1 - C))
    return np.pad(
        im, [(0 if r > (R-1)/2 else rpad, 0 if r < (R-1)/2 else rpad),
             (0 if c > (C-1)/2 else cpad, 0 if c < (C-1)/2 else cpad),
             (0, 0)], 'constant')

def find_centers(p1, p2):
    cx = np.round(np.mean([p1[0], p2[0]]))
    cy = np.round(np.mean([p1[1], p2[1]]))
    return cx, cy

def align_image_centers(im1, im2, pts):
    p1, p2, p3, p4 = pts
    h1, w1, b1 = im1.shape
    h2, w2, b2 = im2.shape
    
    cx1, cy1 = find_centers(p1, p2)
    cx2, cy2 = find_centers(p3, p4)

    im1 = recenter(im1, cy1, cx1)
    im2 = recenter(im2, cy2, cx2)
    return im1, im2

def rescale_images(im1, im2, pts):
    p1, p2, p3, p4 = pts
    len1 = np.sqrt((p2[1] - p1[1])**2 + (p2[0] - p1[0])**2)
    len2 = np.sqrt((p4[1] - p3[1])**2 + (p4[0] - p3[0])**2)
    dscale = len2/len1
    if dscale < 1:
        im1 = sktr.rescale(im1, dscale, channel_axis=2)
    else:
        im2 = sktr.rescale(im2, 1./dscale, channel_axis=2)
    return im1, im2

def rotate_im1(im1, im2, pts):
    p1, p2, p3, p4 = pts
    theta1 = math.atan2(-(p2[1] - p1[1]), (p2[0] - p1[0]))
    theta2 = math.atan2(-(p4[1] - p3[1]), (p4[0] - p3[0]))
    dtheta = theta2 - theta1
    im1 = sktr.rotate(im1, dtheta*180/np.pi)
    return im1, dtheta

def match_img_size(im1, im2):
    # Make images the same size
    h1, w1, c1 = im1.shape
    h2, w2, c2 = im2.shape
    if h1 < h2:
        im2 = im2[int(np.floor((h2-h1)/2.)) : -int(np.ceil((h2-h1)/2.)), :, :]
    elif h1 > h2:
        im1 = im1[int(np.floor((h1-h2)/2.)) : -int(np.ceil((h1-h2)/2.)), :, :]
    if w1 < w2:
        im2 = im2[:, int(np.floor((w2-w1)/2.)) : -int(np.ceil((w2-w1)/2.)), :]
    elif w1 > w2:
        im1 = im1[:, int(np.floor((w1-w2)/2.)) : -int(np.ceil((w1-w2)/2.)), :]
    assert im1.shape == im2.shape
    return im1, im2

def align_images(im1, im2):
    pts = get_points(im1, im2)
    im1, im2 = align_image_centers(im1, im2, pts)
    im1, im2 = rescale_images(im1, im2, pts)
    im1, angle = rotate_im1(im1, im2, pts)
    im1, im2 = match_img_size(im1, im2)
    return im1, im2

#TODO: 
# How do I apply the filter? Do I need to go pixel by pixel style and change them?
# Is low / high fine ?
# Can I use cv2 or the scipy.signal.convolve2d? 
# Also what is the scipy.signal.convolve2d thing used for, how do I use it?
# Do I need gaussian kernal for both low and high filter?
# Once you create your kernel, filtering can be done by scipy.signal.convolve2d - what does this mean?

#Tips for pyramid blending?

def gaussian_kernel(size, sigma):

    # Generates a 2D Gaussian kernel by approximating the continuous Gaussian function hopefully

    ax = np.arange(-size // 2 + 1, size // 2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    return kernel / np.sum(kernel)

# wasn't working on home PC, works on laptop? 
def low_pass_filter(img, kernel_size, sigma):

    # Applies a low-pass filter to the input image using a Gaussian kernel (WIP)

    kernel = gaussian_kernel(kernel_size, sigma)
    # filtered_image = cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)
    filtered_image = cv2.filter2D(img, -1, kernel)
    return filtered_image
    
def high_pass_filter(img, kernel_size, sigma):
    # Apply Gaussian filter to the image
    blurred = cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)
    
    # Subtract the Gaussian-filtered image from the original
    high_pass = img - blurred
    
    return high_pass

if __name__ == "__main__":

    imageDir = '../Images/'
    outDir = '../Results/'

    im1_name = 'bung.jpg'
    im2_name = 'corb2.jpg'

    # 1. load the images
	
	# Low frequency image
    im1 = plt.imread(imageDir + im1_name) # read the input image
    info = np.iinfo(im1.dtype) # get information about the image type (min max values)
    im1 = im1.astype(np.float32) / info.max # normalize the image into range 0 and 1
    
	# High frequency image
    im2 = plt.imread(imageDir + im2_name) # read the input image
    info = np.iinfo(im2.dtype) # get information about the image type (min max values)
    im2 = im2.astype(np.float32) / info.max # normalize the image into range 0 and 1
    
    # 2. align the two images by calling align_images
    im1_aligned, im2_aligned = align_images(im1, im2)
    
    if isGray:
        im1_aligned = np.mean(im1_aligned, axis=2)
        im2_aligned = np.mean(im2_aligned, axis=2)
	
    # Now you are ready to write your own code for creating hybrid images!
    im_low = im1_aligned
    im_low = low_pass_filter(im_low, 20, 20)
    
    im_high = im2_aligned
    im_high = high_pass_filter(im_high, 15, 10)
    im = im_low + im_high

    im = im / im.max()
	
    if isGray:
        plt.imsave(outDir + im1_name[:-4] + '_' + im2_name[:-4] + '_Hybrid.jpg', im, cmap='gray')
    else:
        plt.imsave(outDir + im1_name[:-4] + '_' + im2_name[:-4] + '_Hybrid.jpg', im)
    
    pass
