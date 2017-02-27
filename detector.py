from util import extract_features, slide_window, draw_boxes
import cv2
import matplotlib.pyplot as plt

class Detector(object):
    def __init__(self, classifier, feature_parameters, shape, scaler):
        self._classifier = classifier
        self._feature_parameters = dict(feature_parameters)
        self._shape = shape
        self._scaler = scaler
        del self._feature_parameters['cspace']

    def __call__(self, img, show_plots=False):
        windows = self.get_windows(img)
        hits = []
        for window in windows:
            roi = img[window[0][1]:window[1][1], window[0][0]:window[1][0], :]
            if roi.shape[0:1] != self._shape:
                roi = cv2.resize(roi, self._shape)
            features = extract_features(roi, **self._feature_parameters).reshape(1, -1)
            features = self._scaler.transform(features)
            if self._classifier.predict(features) == 1:
                hits.append(window)
        if show_plots:
            disp = draw_boxes(img, windows, color=(0, 0, 255), thick=6)
            disp = draw_boxes(disp, hits, color=(255, 255, 0), thick=3)
            plt.figure()
            plt.imshow(disp)
        return hits

    def get_windows(self, img):
        height, width, depth = img.shape
        sizes = [(32,  0.5,  [width//3, 2*width//3], [height//2, 3*height//4]),
                 (48,  0.0,  [None, None], [height//2, 3*height//4]),
                 (64,  0.5,  [None, None], [height//2, 3*height//4]),
                 (80,  0.5,  [None, None], [height//2, 3*height//4]),
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
