from util import extract_features, rgb, slide_window, draw_boxes, make_heatmap, get_hog_features, color_hist
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.measurements import label as label_image
from skimage.filters.rank import windowed_histogram

class Detector(object):
    def __init__(self, classifier, feature_parameters, shape, scaler, heat_threshold, alpha=1.0):
        self._alpha = alpha
        self._last_heatmap = None
        self._classifier = classifier
        self._feature_parameters = dict(feature_parameters)
        self._shape = shape
        self._scaler = scaler
        self._threshold = heat_threshold
        self._cspace = self._feature_parameters['cspace']
        del self._feature_parameters['cspace']

    def __call__(self, img, show_plots=False):
        hits = self.get_hits(img)
        return hits
        heat = make_heatmap(img.shape[0:2], hits)
        if self._last_heatmap is None:
            self._last_heatmap = heat
        filtered_heat = (1-self._alpha) * self._last_heatmap + self._alpha * heat
        self._last_heatmap = filtered_heat
        binary = filtered_heat >= self._threshold
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
        pix_per_cell = self._feature_parameters['hog_pix_per_cell']
        x_cells_per_window = self._shape[1] // pix_per_cell - 1
        y_cells_per_window = self._shape[0] // pix_per_cell - 1
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
            # Calculate ROI to avoid processing more than we have to
            roi_x = (int(x_range[0] * img.shape[1]), int(x_range[1] * img.shape[1]))
            roi_y = (int(y_range[0] * img.shape[0]), int(y_range[1] * img.shape[0]))
            roi = img[roi_y[0]:roi_y[1], roi_x[0]:roi_x[1], :]
            # Scale the ROI
            scaled_shape = (int(roi.shape[1] * scale), int(roi.shape[0] * scale))
            scaled_roi = cv2.resize(roi, scaled_shape)
            # Calculate HOG features for whole scaled ROI at once
            hog = [get_hog_features(scaled_roi[:,:,c],
                                orient = self._feature_parameters['hog_orient'],
                                pix_per_cell = self._feature_parameters['hog_pix_per_cell'],
                                cell_per_block = self._feature_parameters['hog_cell_per_block'],
                                feature_vec=False) for c in range(scaled_roi.shape[-1])]
            hog_shape = hog[0].shape
            # Calculate color features for whole scaled ROI at once
            if self._feature_parameters['hist_bins'] > 0:
                histo = [windowed_histogram((scaled_roi[:,:,c]*255).astype(np.uint8),
                        selem=np.ones(self._shape),
                        shift_x = -self._shape[1]/2,
                        shift_y = -self._shape[0]/2,
                        n_bins=self._feature_parameters['hist_bins']) for c in range(scaled_roi.shape[-1])]
            # Calculate bounds for iterating over the HOG feature image
            x_start = 0
            x_stop = hog_shape[1] - x_cells_per_window + 1
            x_step = int((1 - overlap) * x_cells_per_window)
            y_start = 0
            y_stop = hog_shape[0] - y_cells_per_window + 1
            y_step = int((1 - overlap) * y_cells_per_window)
            for x in range(x_start, x_stop, x_step):
                for y in range(y_start, y_stop, y_step):
                    # Extract color features
                    if self._feature_parameters['hist_bins'] > 0:
                        color_features = np.hstack([h[(y * pix_per_cell), (x * pix_per_cell), :].ravel() for h in histo])
                    else:
                        color_features = []
                    # TODO: Add spatial?
                    spatial_features = []
                    # Extract hog features
                    hog_features = np.hstack([h[y:y+y_cells_per_window, x:x+x_cells_per_window].ravel() for h in hog])
                    # Vectorize features
                    features = np.concatenate((spatial_features, color_features, hog_features))
                    features = features.reshape(1, -1)
                    features = self._scaler.transform(features)
                    # Create window (in unscaled image dimensions)
                    window_start = (roi_x[0] + int(x/scale * pix_per_cell), roi_y[0] + int(y/scale * pix_per_cell))
                    window_end = (int(window_start[0] + self._shape[1]/scale), int(window_start[1] + self._shape[0]/scale))
                    # Check if the window is a vehicle
                    if self._classifier.decision_function(features) > 1.0:
                        hits.append((window_start, window_end))
        return hits
