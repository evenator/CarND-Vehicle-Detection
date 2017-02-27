from util import extract_features, rgb, slide_window, draw_boxes, make_heatmap
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

    def __call__(self, img, show_plots=False):
        hits = self.get_hits(img)
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

    def get_windows(self, img):
        height, width, depth = img.shape
        sizes = [(32,  0.5, [width//3, 2*width//3], [height//2, 3*height//4]),
                 (48,  0.0, [None, None], [height//2, 3*height//4]),
                 (64,  0.5, [None, None], [height//2, 3*height//4]),
                 (80,  0.5, [None, None], [height//2, 3*height//4]),
                 (96,  0.5, [None, None], [height//2, 7*height//8]),
                 (112, 0.5, [None, None], [height//2, 7*height//8]),
                 (128, 0.5, [None, None], [height//2, 7*height//8])]
        windows = []
        for size, overlap, x_range, y_range in sizes:
            windows.extend(slide_window(img,
                                        x_start_stop=x_range,
                                        y_start_stop=y_range,
                                        xy_window=(size, size),
                                        xy_overlap=(overlap, overlap)))
        return windows
