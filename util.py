import cv2
import numpy as np
from skimage.feature import hog

'''
This module contains various functions written as part of the lesson exercises
which may be helpful here.
'''

def read_image(path, cspace='RGB'):
    '''
    Load image and convert to 32-bit float [0,1] in the requested colorspace
    '''
    img = cv2.imread(path).astype(np.float32)/255
    if cspace == 'HSV':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    elif cspace == 'HSL':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSL)
    elif cspace == 'RGB':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif cspace == 'LUV':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
    elif cspace == 'YUV':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    elif cspace == 'YCrCb':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    else:
        raise Exception("unknown colorspace " + cspace)
    return img

def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the 3 channels separately
    hist0 = np.histogram(img[:,:,0], nbins, bins_range)
    hist1 = np.histogram(img[:,:,1], nbins, bins_range)
    hist2 = np.histogram(img[:,:,2], nbins, bins_range)
    # Generating bin centers
    bin_centers = hist0[1]
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((hist0[0], hist1[0], hist2[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist0, hist1, hist2, bin_centers, hist_features

def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    output = np.copy(img)
    for bb in bboxes:
        cv2.rectangle(output, bb[0], bb[1], color, thick)
    return output

def bin_spatial(img, size=(32, 32)):
    features = cv2.resize(img, size).ravel()
    # Return the feature vector
    return features

def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False,
                     feature_vec=True):
    return hog(img, orientations=orient,
               pixels_per_cell=(pix_per_cell, pix_per_cell),
               cells_per_block=(cell_per_block, cell_per_block),
               transform_sqrt=True, visualise=vis, feature_vector=feature_vec)

def extract_features_from_images(paths, cspace='RGB', spatial_size=(32, 32),
                     hist_bins=32, hist_range=(0, 256),
                     hog_orient=9, hog_pix_per_cell = 8, hog_cell_per_block=2, hog_channel=0):
    '''
    Extract features from a list of image paths
    '''
    features = []
    for path in paths:
        img = read_image(path, cspace)
        features.append(extract_features(img, spatial_size,
                                        hist_bins, hist_range,
                                        hog_orient, hog_pix_per_cell,
                                        hog_cell_per_block, hog_channel))
    return features

def extract_features(img, spatial_size=(32, 32),
                     hist_bins=32, hist_range=(0, 256),
                     hog_orient=9, hog_pix_per_cell = 8, hog_cell_per_block=2, hog_channel=0):
    '''
    Extract features from a single image
    '''
    spatial_features = bin_spatial(img, spatial_size)
    color_features = color_hist(img, hist_bins, hist_range)
    if hog_channel == 'ALL':
        hog_features = []
        for channel in range(img.shape[2]):
            hog_features.append(get_hog_features(img[:,:,channel],
                                hog_orient, hog_pix_per_cell, hog_cell_per_block, 
                                vis=False, feature_vec=True))
        hog_features = np.ravel(hog_features)        
    elif hog_channel == 'NONE':
        pass
    else:
        hog_features = get_hog_features(img[:,:,hog_channel],
                                        hog_orient, hog_pix_per_cell,
                                        hog_cell_per_block, vis=False,
                                        feature_vec=True)
    features = np.concatenate((spatial_features, color_features, hog_features))
    return features

def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], 
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] is None:
        x_start_stop[0] = 0
    if x_start_stop[1] is None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] is None:
        y_start_stop[0] = 0
    if y_start_stop[1] is None:
        y_start_stop[1] = img.shape[0]
    
    x_step = int(xy_window[0] * (1 - xy_overlap[0]))
    x_start = x_start_stop[0]
    x_stop = x_start_stop[1] - xy_window[0] + 1
    y_step = int(xy_window[1] * (1 - xy_overlap[1]))
    y_start = y_start_stop[0]
    y_stop = y_start_stop[1] - xy_window[1] + 1
    
    window_list = []
    for x in range(x_start, x_stop, x_step):
        for y in range(y_start, y_stop, y_step):
            window_list.append(((x, y), (x + xy_window[0], y + xy_window[1])))
    return window_list