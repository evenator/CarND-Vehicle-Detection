import cv2
import numpy as np
import os
from skimage.feature import hog

'''
This module contains various functions written as part of the lesson exercises
which may be helpful here.
'''

def convert_video_frame(frame, cspace='RGB'):
    '''
    Convert a uint8 RGB video frame to [0,1] float32
    '''
    frame = frame.astype(np.float32)/255
    if cspace == 'HSV':
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    elif cspace == 'HSL':
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2HSL)
    elif cspace == 'RGB':
        pass
    elif cspace == 'LUV':
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2LUV)
    elif cspace == 'YUV':
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2YUV)
    elif cspace == 'YCrCb':
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2YCrCb)
    else:
        raise Exception("unknown colorspace " + cspace)
    return frame


def make_heatmap(shape, bbox_list):
    '''
    Create a uint8 heatmap by incrementing all the pixels inside
    of the given bounding boxes
    '''
    heatmap = np.zeros(shape, dtype=np.float32)
    for item in bbox_list:
        box = item[0], item[1]
        if len(item) > 2:
            heat = item[2]
        else:
            heat = 1
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += heat
    return heatmap


def rgb(src, src_cspace):
    '''
    Convert a 32 bit float [0,1] image to a uint8 image in the rgb colorspace
    '''
    if src_cspace == 'HSV':
        img = cv2.cvtColor(src, cv2.COLOR_HSV2RGB)
    elif src_cspace == 'HSL':
        img = cv2.cvtColor(src, cv2.COLOR_HSL2RGB)
    elif src_cspace == 'RGB':
        img = cv2.cvtColor(src, cv2.COLOR_RGB2RGB)
    elif src_cspace == 'LUV':
        img = cv2.cvtColor(src, cv2.COLOR_LUV2RGB)
    elif src_cspace == 'YUV':
        img = cv2.cvtColor(src, cv2.COLOR_YUV2RGB)
    elif src_cspace == 'YCrCb':
        img = cv2.cvtColor(src, cv2.COLOR_YCrCb2RGB)
    else:
        raise Exception("unknown colorspace " + cspace)
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)
    return img


def write_image(path, src, src_cspace):
    '''
    Write an image to a file. Input can be 32-bit float [0,1] or uint8
    '''
    if src_cspace == 'HSV':
        img = cv2.cvtColor(src, cv2.COLOR_HSV2BGR)
    elif src_cspace == 'HSL':
        img = cv2.cvtColor(src, cv2.COLOR_HSL2BGR)
    elif src_cspace == 'RGB':
        img = cv2.cvtColor(src, cv2.COLOR_RGB2BGR)
    elif src_cspace == 'LUV':
        img = cv2.cvtColor(src, cv2.COLOR_LUV2RGB)
    elif src_cspace == 'YUV':
        img = cv2.cvtColor(src, cv2.COLOR_YUV2BGR)
    elif src_cspace == 'YCrCb':
        img = cv2.cvtColor(src, cv2.COLOR_YCrCb2BGR)
    else:
        raise Exception("unknown colorspace " + cspace)
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)
    cv2.imwrite(path, img)


def read_image(path, cspace='RGB'):
    '''
    Load image and convert to 32-bit float [0,1] in the requested colorspace
    '''
    if not os.path.exists(path):
        raise IOError(path + " does not exist or could not be loaded")
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
    ''' Compute the normalized color histogram'''
    n_pix = img.shape[0] * img.shape[1]
    return np.concatenate([np.histogram(img[:,:,c], nbins, bins_range)[0] for c in range(img.shape[-1])])/n_pix


def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    '''
    Draw bounding boxes on an image
    '''
    output = np.copy(img)
    for bb in bboxes:
        if len(bb) > 2:
            color = np.array(color) * bb[2]
        cv2.rectangle(output, bb[0], bb[1], color, thick)
    return output


def bin_spatial(img, size=(32, 32)):
    '''
    Scaled the image patch to specified size, then vectorize it to use as a
    feature
    '''
    return cv2.resize(img, size).ravel()


def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False,
                     feature_vec=True):
    '''
    Convenience function that wraps skimage's hog function
    '''
    return hog(img, orientations=orient,
               pixels_per_cell=(pix_per_cell, pix_per_cell),
               cells_per_block=(cell_per_block, cell_per_block),
               transform_sqrt=True, visualise=vis, feature_vector=feature_vec)


def extract_features_from_images(paths, cspace='RGB', spatial_size=None,
                     hist_bins=0,
                     hog_orient=9, hog_pix_per_cell = 8, hog_cell_per_block=2, hog_channel=0):
    '''
    Extract features from a list of image paths
    '''
    features = []
    for path in paths:
        img = read_image(path, cspace)
        features.append(extract_features(img, spatial_size,
                                        hist_bins,
                                        hog_orient, hog_pix_per_cell,
                                        hog_cell_per_block, hog_channel))
    return np.array(features)


def extract_features(img, spatial_size=(32, 32),
                     hist_bins=0,
                     hog_orient=9, hog_pix_per_cell = 8, hog_cell_per_block=2, hog_channel=0):
    '''
    Extract features from a single image
    '''
    spatial_features = []
    if spatial_size is not None:
        spatial_features = bin_spatial(img, spatial_size)
    color_features = []
    if hist_bins > 0:
        color_features = color_hist(img*255, hist_bins)
    if hog_channel == 'ALL':
        hog_features = []
        for channel in range(img.shape[2]):
            hog_features.append(get_hog_features(img[:,:,channel],
                                hog_orient, hog_pix_per_cell, hog_cell_per_block,
                                vis=False, feature_vec=True))
        hog_features = np.ravel(hog_features)
    elif hog_channel == 'NONE':
        hog_features = []
    else:
        hog_features = get_hog_features(img[:,:,hog_channel],
                                        hog_orient, hog_pix_per_cell,
                                        hog_cell_per_block, vis=False,
                                        feature_vec=True)
    features = np.concatenate((spatial_features, color_features, hog_features))
    return features


def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None],
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    '''
    Generate a list of windows by sliding them over the given image
    '''
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

