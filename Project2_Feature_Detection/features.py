import math

import cv2
import numpy as np
import scipy
from scipy import ndimage, spatial

import transformations


def inbounds(shape, indices):
    assert len(shape) == len(indices)
    for i, ind in enumerate(indices):
        if ind < 0 or ind >= shape[i]:
            return False
    return True


## Keypoint detectors ##########################################################

class KeypointDetector(object):
    def detectKeypoints(self, image):
        '''
        Input:
            image -- uint8 BGR image with values between [0, 255]
        Output:
            list of detected keypoints, fill the cv2.KeyPoint objects with the
            coordinates of the detected keypoints, the angle of the gradient
            (in degrees), the detector response (Harris score for Harris detector)
            and set the size to 10.
        '''
        raise NotImplementedError()


class DummyKeypointDetector(KeypointDetector):
    '''
    Compute silly example features. This doesn't do anything meaningful, but
    may be useful to use as an example.
    '''

    def detectKeypoints(self, image):
        '''
        Input:
            image -- uint8 BGR image with values between [0, 255]
        Output:
            list of detected keypoints, fill the cv2.KeyPoint objects with the
            coordinates of the detected keypoints, the angle of the gradient
            (in degrees), the detector response (Harris score for Harris detector)
            and set the size to 10.
        '''
        image = image.astype(np.float32)
        image /= 255.
        features = []
        height, width = image.shape[:2]

        for y in range(height):
            for x in range(width):
                r = image[y, x, 0]
                g = image[y, x, 1]
                b = image[y, x, 2]

                if int(255 * (r + g + b) + 0.5) % 100 == 1:
                    # If the pixel satisfies this meaningless criterion,
                    # make it a feature.

                    f = cv2.KeyPoint()
                    f.pt = (x, y)
                    # Dummy size
                    f.size = 10
                    f.angle = 0
                    f.response = 10

                    features.append(f)

        return features


class HarrisKeypointDetector(KeypointDetector):

    def saveHarrisImage(self, harrisImage, srcImage):
        '''
        Saves a visualization of the harrisImage, by overlaying the harris
        response image as red over the srcImage.

        Input:
            srcImage -- Grayscale input image in a numpy array with
                        values in [0, 1]. The dimensions are (rows, cols).
            harrisImage -- Grayscale input image in a numpy array with
                        values in [0, 1]. The dimensions are (rows, cols).
        '''
        outshape = [harrisImage.shape[0], harrisImage.shape[1], 3]
        outImage = np.zeros(outshape)
        # Make a grayscale srcImage as a background
        srcNorm = srcImage * (0.3 * 255 / (np.max(srcImage) + 1e-50))
        outImage[:, :, :] = np.expand_dims(srcNorm, 2)

        # Add in the harris keypoints as red
        outImage[:, :, 2] += harrisImage * (4 * 255 / (np.max(harrisImage)) + 1e-50)
        cv2.imwrite("harris.png", outImage)

    # Compute harris values of an image.
    def computeHarrisValues(self, srcImage):
        '''
        Input:
            srcImage -- Grayscale input image in a numpy array with
                        values in [0, 1]. The dimensions are (rows, cols).
        Output:
            harrisImage -- numpy array containing the Harris score at
                           each pixel.
            orientationImage -- numpy array containing the orientation of the
                                gradient at each pixel in degrees.
        '''
        height, width = srcImage.shape[:2]

        harrisImage = np.zeros(srcImage.shape[:2])
        orientationImage = np.zeros(srcImage.shape[:2])

        # Use the 3x3 Sobel operator to compute the x, y derivatives
        dx = ndimage.sobel(srcImage, 0)
        dy = ndimage.sobel(srcImage, 1)

        # Use a 5x5 Gaussian mask with 0.5 sigma for weights
        A = ndimage.gaussian_filter(dx * dx, sigma = 0.5)
        B = ndimage.gaussian_filter(dx * dy, sigma = 0.5)
        C = ndimage.gaussian_filter(dy * dy, sigma = 0.5)

        det_H = A * C - B ** 2
        trace_H = A + C

        # Compute harris corner strength for 'srcImage' at each pixel  
        harrisImage = det_H - 0.1 * trace_H ** 2
        # Compute orientation for each pixel and store in 'orientationImage.'
        orientationImage = np.degrees(np.arctan2(dx, dy))

        # Save the harris image as harris.png for the website assignment
        self.saveHarrisImage(harrisImage, srcImage)

        return harrisImage, orientationImage

    def computeLocalMaxima(self, harrisImage):
        '''
        Input:
            harrisImage -- numpy array containing the Harris score at
                           each pixel.
        Output:
            destImage -- numpy array containing True/False at
                         each pixel, depending on whether
                         the pixel value is the local maxima in
                         its 7x7 neighborhood.
        '''
        destImage = np.zeros_like(harrisImage, np.bool)

        # Compute the local maxima image
        local_max = ndimage.maximum_filter(harrisImage, size = (7, 7))
        destImage = (harrisImage == local_max)

        return destImage

    def detectKeypoints(self, image):
        '''
        Input:
            image -- BGR image with values between [0, 255]
        Output:
            list of detected keypoints, fill the cv2.KeyPoint objects with the
            coordinates of the detected keypoints, the angle of the gradient
            (in degrees), the detector response (Harris score for Harris detector)
            and set the size to 10.
        '''
        image = image.astype(np.float32)
        image /= 255.
        height, width = image.shape[:2]
        features = []

        # Create grayscale image used for Harris detection
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # computeHarrisValues() computes the harris score at each pixel
        # position, storing the result in harrisImage.
        # You will need to implement this function.
        harrisImage, orientationImage = self.computeHarrisValues(grayImage)

        # Compute local maxima in the Harris image.  You will need to
        # implement this function. Create image to store local maximum harris
        # values as True, other pixels False
        harrisMaxImage = self.computeLocalMaxima(harrisImage)

        # Loop through feature points in harrisMaxImage and fill in information
        # needed for descriptor computation for each point.
        # You need to fill x, y, and angle.
        for y in range(height):
            for x in range(width):
                if not harrisMaxImage[y, x]:
                    continue

                f = cv2.KeyPoint()

                # Fill in feature f with location and orientation
                # Set f.size to 10, f.pt to the (x,y) coordinate
                # f.angle to the orientation in degrees
                # f.response to the harris score
                f.size = 10
                f.pt = (x, y)
                f.angle = orientationImage[y, x]
                f.response = harrisImage[y, x]
                features.append(f)

        return features


class ORBKeypointDetector(KeypointDetector):
    def detectKeypoints(self, image):
        '''
        Input:
            image -- uint8 BGR image with values between [0, 255]
        Output:
            list of detected keypoints, fill the cv2.KeyPoint objects with the
            coordinates of the detected keypoints, the angle of the gradient
            (in degrees) and set the size to 10.
        '''
        detector = cv2.ORB()
        return detector.detect(image)


## Feature descriptors #########################################################


class FeatureDescriptor(object):
    # Implement in child classes
    def describeFeatures(self, image, keypoints):
        '''
        Input:
            image -- BGR image with values between [0, 255]
            keypoints -- the detected features, we have to compute the feature
            descriptors at the specified coordinates
        Output:
            Descriptor numpy array, dimensions:
                keypoint number x feature descriptor dimension
        '''
        raise NotImplementedError


class SimpleFeatureDescriptor(FeatureDescriptor):

    def describeFeatures(self, image, keypoints):
        '''
        Input:
            image -- BGR image with values between [0, 255]
            keypoints -- the detected features, we have to compute the feature
                         descriptors at the specified coordinates
        Output:
            desc -- K x 25 numpy array, where K is the number of keypoints
        '''
        image = image.astype(np.float32)
        image /= 255.
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        desc = np.zeros((len(keypoints), 5 * 5))

        height = image.shape[0]
        width = image.shape[1]

        # initialize canvas and center image
        pad = np.zeros((height + 5, width + 5))
        pad[2: 2 + height, 2: 2 + width] = grayImage
        for i, f in enumerate(keypoints):
            x, y = f.pt
            x, y = int(x), int(y)

            # The simple descriptor is a 5x5 window of intensities
            # sampled centered on the feature point. Store the descriptor
            # as a row-major vector. Treat pixels outside the image as zero.
            window = pad[y : y + 5, x : x + 5]
            desc[i, :] = window.reshape((25,))

        return desc


class MOPSFeatureDescriptor(FeatureDescriptor):
    def get_rot_mx(angle):
        '''
        Input:
            angle_x -- Rotation around the x axis in radians
            angle_y -- Rotation around the y axis in radians
            angle_z -- Rotation around the z axis in radians
        Output:
            A 4x4 numpy array representing 3D rotations. The order of the rotation
            axes from first to last is x, y, z, if you multiply with the resulting
            rotation matrix from left.
        '''
        # Note: For MOPS, you need to use angle_z only, since we are in 2D

        rot_mx = np.array([[math.cos(angle), -math.sin(angle), 0],
                             [math.sin(angle), math.cos(angle), 0],
                             [0, 0, 1]])

        return rot_mx


    def get_trans_mx(trans_vec):
        '''
        Input:
            trans_vec -- Translation vector represented by an 1D numpy array with 2
            elements
        Output:
            A 4x4 numpy array representing 3D translation.
        '''
        assert trans_vec.ndim == 1
        assert trans_vec.shape[0] == 2

        trans_mx = np.eye(3)
        trans_mx[:2, 2] = trans_vec

        return trans_mx


    def get_scale_mx(s):
        '''
        Input:
            s_x -- Scaling along the x axis
            s_y -- Scaling along the y axis
            s_z -- Scaling along the z axis
        Output:
            A 4x4 numpy array representing 3D scaling.
        '''
        # Note: For MOPS, you need to use s_x and s_y only, since we are in 2D

        scale_mx = np.eye(3)

        for i in range(2):
            scale_mx[i, i] = s

        return scale_mx


    # TODO: Implement parts of this function
    def describeFeatures(self, image, keypoints):
        '''
        Input:
            image -- BGR image with values between [0, 255]
            keypoints -- the detected features, we have to compute the feature
            descriptors at the specified coordinates
        Output:
            desc -- K x W^2 numpy array, where K is the number of keypoints
                    and W is the window size
        '''
        image = image.astype(np.float32)
        image /= 255.
        # This image represents the window around the feature you need to
        # compute to store as the feature descriptor (row-major)
        windowSize = 8
        desc = np.zeros((len(keypoints), windowSize * windowSize))
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        grayImage = ndimage.gaussian_filter(grayImage, 0.5)

        for i, f in enumerate(keypoints):
            # Compute the transform as described by the feature
            # location/orientation. You will need to compute the transform
            # from each pixel in the 40x40 rotated window surrounding
            # the feature to the appropriate pixels in the 8x8 feature
            # descriptor image.
            transMx = np.zeros((2, 3))

            # Get feature x, y
            x, y = f.pt

            # compute translation matrix around point f
            trans_vec = np.array([-x, -y])
            T1 = get_trans_mx(trans_vec)

            # Compute rotation matrix
            rot_angle = - f.angle / 180. * math.pi
            R = get_rot_mx(rot_angle)

            # Compute scaling matrix by a factor of 1/5
            S = get_scale_mx(1./5.)

            # compute 2nd translation matrix -- move to top left corner
            trans_vec2 = np.array([4, 4])
            T2 = get_trans_mx(trans_vec2)

            trans_matrix = np.dot(np.dot(np.dot(T2, S), R), T1)

            transMx = trans_matrix[0:2, 0:3]

            # Call the warp affine function to do the mapping
            # It expects a 2x3 matrix
            destImage = cv2.warpAffine(grayImage, transMx,
                (windowSize, windowSize), flags=cv2.INTER_LINEAR)

            # Normalize the descriptor to have zero mean and unit
            # variance. If the variance is zero then set the descriptor
            # vector to zero. Lastly, write the vector to desc.
            z_mean = destImage - np.mean(destImage)
            dev = np.std(z_mean)

            if (dev <= 10**(-5)) :
                desc[i, :] = np.zeros(windowSize * windowSize)
            else :
                norm = z_mean / dev
                desc[i, :] = norm.reshape(windowSize * windowSize)

        return desc


class ORBFeatureDescriptor(KeypointDetector):
    def describeFeatures(self, image, keypoints):
        '''
        Input:
            image -- BGR image with values between [0, 255]
            keypoints -- the detected features, we have to compute the feature
            descriptors at the specified coordinates
        Output:
            Descriptor numpy array, dimensions:
                keypoint number x feature descriptor dimension
        '''
        descriptor = cv2.ORB()
        kps, desc = descriptor.compute(image, keypoints)
        if desc is None:
            desc = np.zeros((0, 128))

        return desc


# Compute Custom descriptors (extra credit)
class CustomFeatureDescriptor(FeatureDescriptor):
    def describeFeatures(self, image, keypoints):
        '''
        Input:
            image -- BGR image with values between [0, 255]
            keypoints -- the detected features, we have to compute the feature
            descriptors at the specified coordinates
        Output:
            Descriptor numpy array, dimensions:
                keypoint number x feature descriptor dimension
        '''
        raise NotImplementedError('NOT IMPLEMENTED')


## Feature matchers ############################################################


class FeatureMatcher(object):
    def matchFeatures(self, desc1, desc2):
        '''
        Input:
            desc1 -- the feature descriptors of image 1 stored in a numpy array,
                dimensions: rows (number of key points) x
                columns (dimension of the feature descriptor)
            desc2 -- the feature descriptors of image 2 stored in a numpy array,
                dimensions: rows (number of key points) x
                columns (dimension of the feature descriptor)
        Output:
            features matches: a list of cv2.DMatch objects
                How to set attributes:
                    queryIdx: The index of the feature in the first image
                    trainIdx: The index of the feature in the second image
                    distance: The distance between the two features
        '''
        raise NotImplementedError

    # Evaluate a match using a ground truth homography.  This computes the
    # average SSD distance between the matched feature points and
    # the actual transformed positions.
    @staticmethod
    def evaluateMatch(features1, features2, matches, h):
        d = 0
        n = 0

        for m in matches:
            id1 = m.queryIdx
            id2 = m.trainIdx
            ptOld = np.array(features2[id2].pt)
            ptNew = FeatureMatcher.applyHomography(features1[id1].pt, h)

            # Euclidean distance
            d += np.linalg.norm(ptNew - ptOld)
            n += 1

        return d / n if n != 0 else 0

    # Transform point by homography.
    @staticmethod
    def applyHomography(pt, h):
        x, y = pt
        d = h[6]*x + h[7]*y + h[8]

        return np.array([(h[0]*x + h[1]*y + h[2]) / d,
            (h[3]*x + h[4]*y + h[5]) / d])


class SSDFeatureMatcher(FeatureMatcher):
    def matchFeatures(self, desc1, desc2):
        '''
        Input:
            desc1 -- the feature descriptors of image 1 stored in a numpy array,
                dimensions: rows (number of key points) x
                columns (dimension of the feature descriptor)
            desc2 -- the feature descriptors of image 2 stored in a numpy array,
                dimensions: rows (number of key points) x
                columns (dimension of the feature descriptor)
        Output:
            features matches: a list of cv2.DMatch objects
                How to set attributes:
                    queryIdx: The index of the feature in the first image
                    trainIdx: The index of the feature in the second image
                    distance: The distance between the two features
        '''
        matches = []
        # feature count = n
        assert desc1.ndim == 2
        # feature count = m
        assert desc2.ndim == 2
        # the two features should have the type
        assert desc1.shape[1] == desc2.shape[1]

        if desc1.shape[0] == 0 or desc2.shape[0] == 0:
            return []

        # Perform simple feature matching.  This uses the SSD
        # distance between two feature vectors, and matches a feature in
        # the first image with the closest feature in the second image.
        # Note: multiple features from the first image may match the same
        # feature in the second image.
        
        dist = spatial.distance.cdist(desc1, desc2, 'euclidean')

        for i in range(dist.shape[0]):
            j = np.argmin(dist[i])
            match = cv2.DMatch()
            match.queryIdx = i
            match.trainIdx = j
            match.distance = dist[i, j]
            matches.append(match)

        return matches


class RatioFeatureMatcher(FeatureMatcher):
    def matchFeatures(self, desc1, desc2):
        '''
        Input:
            desc1 -- the feature descriptors of image 1 stored in a numpy array,
                dimensions: rows (number of key points) x
                columns (dimension of the feature descriptor)
            desc2 -- the feature descriptors of image 2 stored in a numpy array,
                dimensions: rows (number of key points) x
                columns (dimension of the feature descriptor)
        Output:
            features matches: a list of cv2.DMatch objects
                How to set attributes:
                    queryIdx: The index of the feature in the first image
                    trainIdx: The index of the feature in the second image
                    distance: The ratio test score
        '''
        matches = []
        # feature count = n
        assert desc1.ndim == 2
        # feature count = m
        assert desc2.ndim == 2
        # the two features should have the type
        assert desc1.shape[1] == desc2.shape[1]

        if desc1.shape[0] == 0 or desc2.shape[0] == 0:
            return []

        # Perform ratio feature matching.
        # This uses the ratio of the SSD distance of the two best matches
        # and matches a feature in the first image with the closest feature in the
        # second image.
        # Note: multiple features from the first image may match the same
        # feature in the second image.
        # You don't need to threshold matches in this function
        dist = spatial.distance.cdist(desc1, desc2, 'euclidean')

        for i in range(dist.shape[0]):
            sorted_idx = np.argsort(dist[i])
            match = cv2.DMatch()
            match.queryIdx = i
            match.trainIdx = sorted_idx[0]
            match.distance = dist[i, sorted_idx[0]] / dist[i, sorted_idx[1]]
            matches.append(match)

        return matches


class ORBFeatureMatcher(FeatureMatcher):
    def __init__(self):
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        super(ORBFeatureMatcher, self).__init__()

    def matchFeatures(self, desc1, desc2):
        return self.bf.match(desc1.astype(np.uint8), desc2.astype(np.uint8))

