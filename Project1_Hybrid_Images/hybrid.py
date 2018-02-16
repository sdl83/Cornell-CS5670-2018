import sys
sys.path.append('/Users/kb/bin/opencv-3.1.0/build/lib/')

import cv2
import math
import numpy as np

def correl(img, kernel):
    ''' Helper function for the correlation function so that we can run separately
    on the RGB feeds'''

    # img dimensions
    height = img.shape[0]
    width = img.shape[1]

    # kernel dimensions
    m, n = kernel.shape

    img_result = np.zeros((height,width))

    # computing a padded image for kernel cross correlation to edges
    kernel_y = m / 2
    kernel_x = n / 2 

    padded = np.pad(img,[(kernel_y,kernel_y),(kernel_x, kernel_x)],'constant')

    kernel = kernel.reshape(-1)
    
    # computing new pixel value
    for h in range(height):

        for w in range(width):

            neighborhood = padded[h:h+m,w:w+n] 
            neigh_matrix = np.reshape(neighborhood, n*m)
            # prod = neighborhood * kernel
            # img_result[h,w] = np.sum(prod)
            img_result[h,w] = np.dot(kernel,neigh_matrix)

    return img_result

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
    img_dim = len(img.shape)

    cross = np.zeros(img.shape)

    # if greyscale, run helper once
    if img_dim == 2:
        cross = correl(img,kernel)

    # else run on each feed
    if img_dim == 3:

        # passing in each of the RGB feeds separately
        for i in range(img_dim):

            cross[:,:,i] = correl(img[:,:,i], kernel)

    return cross


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

    kernel = np.zeros((height, width))

    # iterating through the height and width of image
    for h in range(height):

        for w in range(width):

            x = w - width/2
            y = - h + height/2
                
            # Gaussian distribution
            kernel[h,w] = 1.0/(2.0 * math.pi * sigma * sigma) * math.exp((-1.0) * ((x * x) + (y * y)) / (2.0 * sigma * sigma))

    # nomalizing by dividing by sum
    normalize = kernel / np.sum(kernel)

    return normalize


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


