import sys
sys.path.append('/Users/kb/bin/opencv-3.1.0/build/lib/')

import cv2
import math
import numpy as np

def correl(img, kernel):
    ''' Helper function for the correlation function so that we can run separately
    on the RGB feeds'''

    # Kernell Dimensions
    m, n = kernel.shape

    # create output dimensions
    img_row = img.shape[0]
    img_col = img.shape[1]
    output_img = np.zeros((img_row,img_col))

    # Pad the img with 0s
    pad_width = kernel.shape[1] / 2   # kernel width floor div by 2
    pad_height = kernel.shape[0] / 2  # kernel width floor div by 2
    pad_img = np.pad(img,[(pad_height,pad_height),(pad_width,pad_width)],'constant',constant_values=(0))

    for r in range(img_row):
        for c in range(img_col):

            neighb_arr = pad_img[r:r+m,c:c+n]  # grab neighs from pad_img
            product_arr = neighb_arr * kernel  # mult with kernel
            output_img[r,c] = product_arr.sum()  # sum and assign to output_img

    return output_img

def cross_correlation_2d(img, kernel):
    '''Given a kernel of arbitrary m x n dimensions, with both m and n being
    odd, compute the cross correlation of the given image with the given
    kernel, such that the output is of the same dimensions as the image and that
    you assume the pixels out of the bounds of the image to be zero. Note that
    you need to apply the kernel to each channel separately, if the given image
    is an RGB image.

    Inputs:
        img:    Either an RGB image (height x width x 3) or a grayscale image
                (height x width) as a numpy array.
        kernel: A 2D numpy array (m x n), with m and n both odd (but may not be
                equal).

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    # getting input size to distinguish between RGB and greyscale images
    input_dim = len(img.shape)

    output = np.zeros(img.shape)

    # if greyscale, run helper once
    if input_dim == 2:
        output = correl(img,kernel)

    # else run on each feed
    if input_dim == 3:

        # passing in each of the RGB feeds  separately
        for i in range(3):
            output[:,:,i] = correl(img[:,:,i], kernel)

    return output


def convolve_2d(img, kernel):
    '''Use cross_correlation_2d() to carry out a 2D convolution.

    Inputs:
        img:    Either an RGB image (height x width x 3) or a grayscale image
                (height x width) as a numpy array.
        kernel: A 2D numpy array (m x n), with m and n both odd (but may not be
                equal).

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    # flipping the kernel
    return cross_correlation_2d(img, kernel[::-1,::-1])

def gaussian_blur_kernel_2d(sigma, height, width):
    '''Return a Gaussian blur kernel of the given dimensions and with the given
    sigma. Note that width and height are different.

    Input:
        sigma:  The parameter that controls the radius of the Gaussian blur.
                Note that, in our case, it is a circular Gaussian (symmetric
                across height and width).
        width:  The width of the kernel.
        height: The height of the kernel.

    Output:
        Return a kernel of dimensions width x height such that convolving it
        with an image results in a Gaussian-blurred image.
    '''
    # x, y = width/2, height/2
    # x1,y1 = x+1, y+1
    # X = np.arange(-x, x + 1, 1.0)**2
    # Y = np.arange(-y,y1, 1.0)**2

    # X = np.exp(-X/(2 * sigma * sigma))
    # Y = np.exp(-Y/(2 * sigma * sigma)) / (2 * sigma * sigma * np.pi)
    # output = np.outer(X,Y)
    
    # normalize = np.sum(Y) * np.sum(X)
    # return output / normalize

    kernel = np.zeros((height, width))

    # iterating through the height and width of image
    for h in range(height):

        for w in range(width):

            x = w - width/2
            y = -h + height/2
                
            # Gaussian distribution
            first = 1/(2 * math.pi * sigma * sigma)
            second = math.exp(-1 * (x * x + y * y) / (2 * sigma * sigma))
            kernel[h,w] = first * second
            
            sum = np.sum(kernel)
            # nomalizing by dividing by sum
            normalize = kernel / sum

    return normalize

    # kernel = np.zeros((height,width))

    # for r in range(height):
    #     for c in range(width):

    #         # translate row and col to x, y coordinates
    #         x = c-width/2
    #         y = -r+height/2

    #         first = 1/(2*math.pi*math.pow(sigma,2))
    #         second = math.exp(-1 * (math.pow(x, 2) + math.pow(y, 2)) / (2 * math.pow(sigma, 2)))
    #         gauss = first*second
    #         kernel[r,c] = gauss

    #         sum = np.sum(kernel)
    #         n_kernel = kernel/sum

    # return n_kernel


def low_pass(img, sigma, size):
    '''Filter the image as if its filtered with a low pass filter of the given
    sigma and a square kernel of the given size. A low pass filter supresses
    the higher frequency components (finer details) of the image.

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    ker = gaussian_blur_kernel_2d(sigma, size, size)

    return convolve_2d(img, ker)


def high_pass(img, sigma, size):
    '''Filter the image as if its filtered with a high pass filter of the given
    sigma and a square kernel of the given size. A high pass filter suppresses
    the lower frequency components (coarse details) of the image.

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    # opposite of low-pass filter -- remove the coarse details
    return img - low_pass(img, sigma, size)


def create_hybrid_image(img1, img2, sigma1, size1, high_low1, sigma2, size2,
        high_low2, mixin_ratio):
    '''This function adds two images to create a hybrid image, based on
    parameters specified by the user.'''
    high_low1 = high_low1.lower()
    high_low2 = high_low2.lower()

    if img1.dtype == np.uint8:
        img1 = img1.astype(np.float32) / 255.0
        img2 = img2.astype(np.float32) / 255.0

    if high_low1 == 'low':
        img1 = low_pass(img1, sigma1, size1)
    else:
        img1 = high_pass(img1, sigma1, size1)

    if high_low2 == 'low':
        img2 = low_pass(img2, sigma2, size2)
    else:
        img2 = high_pass(img2, sigma2, size2)

    img1 *= 2 * (1 - mixin_ratio)
    img2 *= 2 * mixin_ratio
    hybrid_img = (img1 + img2)
    return (hybrid_img * 255).clip(0, 255).astype(np.uint8)


