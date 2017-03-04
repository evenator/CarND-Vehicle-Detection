from util import extract_features, rgb, slide_window, draw_boxes, make_heatmap, get_hog_features, color_hist
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.measurements import label as label_image

class Detector(object):
    def __init__(self, classifier, feature_parameters, shape, scaler, heat_threshold):
        self._classifier = classifier
        self._feature_parameters = dict(feature_parameters)
        self._shape = shape
        self._scaler = scaler
        self._threshold = heat_threshold
        self._cspace = self._feature_parameters['cspace']
        del self._feature_parameters['cspace']

    def __call__(self, img, show_plots=True):
        hits = self.get_hits_faster(img)
        heat = make_heatmap(img.shape[0:2], hits)
        binary = heat >= self._threshold
        labels = label_image(binary)
        boxes = []
        for i in range(labels[1]):
            y_points, x_points = np.where(labels[0] == i+1)
            boxes.append(((np.min(x_points), np.min(y_points)),
                          (np.max(x_points), np.max(y_points))))
        if show_plots:
            f, ((a0, a1), (a2, a3)) = plt.subplots(2, 2)
            a0.set_title('Raw Hits')
            a0.imshow(draw_boxes(rgb(img, self._cspace), hits))
            a1.set_title('Heatmap')
            a1.imshow(heat.astype(np.float32)/np.max(heat), cmap='gray')
            a2.set_title('Thresholded Heatmap')
            a2.imshow(binary, cmap='gray')
            a3.set_title('Label Image')
            a3.imshow(labels[0], cmap='gray')
        return boxes

    def get_hits(self, img):
        windows = self.get_windows(img)
        hits = []
        for window in windows:
            roi = img[window[0][1]:window[1][1], window[0][0]:window[1][0], :]
            if roi.shape[0:2] != self._shape:
                roi = cv2.resize(roi, self._shape)
            features = extract_features(roi, **self._feature_parameters).reshape(1, -1)
            features = self._scaler.transform(features)
            if self._classifier.predict(features) == 1:
                hits.append(window)
        return hits

    def get_hits_faster(self, img):
        pix_per_cell = self._feature_parameters['hog_pix_per_cell']
        scales = [
#                 (2.0, 0.0, [ 1/3,  2/3], [.5, .75]),
#                 (4/3, 0.0, [0, 1], [.5, .75]),
#                 (1.0, 0.5, [0, 1], [.5, .75]),
#                 (0.8, 0.5, [0, 1], [.5, .75]),
                 (2/3, 0.75, [0, 1], [.5, .875]),
#                 (4/7, 0.75, [0, 1], [.5, .875]),
                 (0.5, 0.75, [0, 1], [.5, .875]),
                 (0.25, 0.75, [0, 1], [.5, .875])
        ]
        hits = []
        for scale, overlap, x_range, y_range in scales:
            new_shape = (int(img.shape[1] * scale), int(img.shape[0] * scale))
            scaled_img = cv2.resize(img, new_shape)
            hog = [get_hog_features(
                                scaled_img[:,:,c],
                                orient = self._feature_parameters['hog_orient'],
                                pix_per_cell = self._feature_parameters['hog_pix_per_cell'],
                                cell_per_block = self._feature_parameters['hog_cell_per_block'],
                                feature_vec=False) for c in range(scaled_img.shape[2])]
            hog_shape = hog[0].shape

            x_block = self._shape[1] // pix_per_cell - 1
            y_block = self._shape[0] // pix_per_cell - 1

            x_start = int(x_range[0]*hog_shape[1])
            x_stop = int(x_range[1]*hog_shape[1]) - x_block + 1
            x_step = int((1 - overlap) * x_block)
            y_start = int(y_range[0]*hog_shape[0])
            y_stop = int(y_range[1]*hog_shape[0]) - y_block + 1
            y_step = int((1 - overlap) * y_block)
            for x in range(x_start, x_stop, x_step):
                for y in range(y_start, y_stop, y_step):
                    # TODO: Add spatial?
                    window_start = (int(x/scale * pix_per_cell), int(y/scale * pix_per_cell))
                    window_end = (int(window_start[0] + self._shape[1]/scale), int(window_start[1] + self._shape[0]/scale))
                    hog_features = np.hstack([h[y:y+y_block, x:x+x_block].ravel() for h in hog])
                    color_features = []
                    spatial_features = []
                    roi = img[window_start[1]:window_end[1], window_start[0]:window_end[0], :]
                    if self._feature_parameters['hist_bins'] > 0:
                        color_features = color_hist(roi, self._feature_parameters['hist_bins'], self._feature_parameters['hist_range'])[-1]
                    features = np.concatenate((spatial_features, color_features, hog_features))
                    features = features.reshape(1, -1)
                    features = self._scaler.transform(features)
                    if self._classifier.decision_function(features) > 1.0:
                        hits.append((window_start, window_end))
        return hits

    def get_windows(self, img):
        height, width, depth = img.shape
        sizes = [
#                 (32,  0.5, [width//3, 2*width//3], [height//2, 3*height//4]),
#                 (48,  0.0, [None, None], [height//2, 3*height//4]),
#                 (64,  0.5, [None, None], [height//2, 3*height//4]),
#                 (80,  0.5, [None, None], [height//2, 3*height//4]),
#                 (96,  0.5, [None, None], [height//2, 7*height//8]),
#                 (112, 0.5, [None, None], [height//2, 7*height//8]),
                 (128, 0.75, [None, None], [height//2, 7*height//8])]
        windows = []
        for size, overlap, x_range, y_range in sizes:
            windows.extend(slide_window(img,
                                        x_start_stop=x_range,
                                        y_start_stop=y_range,
                                        xy_window=(size, size),
                                        xy_overlap=(overlap, overlap)))
        return windows
