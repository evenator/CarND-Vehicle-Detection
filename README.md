Self Driving Car Nanodegree Project 5: Vehicle Detection
========================================================

In this project, I wrote an image processing pipeline to detect vehicles in
still images and video. The general process is as follows:

1. Train a binary image classifier using Histogram of Oriented Gradients (HOG)
  features from from vehicle and non-vehicle example images.
2. Use a sliding window technique to search for vehicles in images at multiple
  scales.
3. Create a heat map of the detected vehicle bounding boxes and estimate a
  single bounding box for each vehicle based on the heatmap.
4. Extend the pipeline to a video stream, persisting the heatmaps from frame
  to frame with decay for smoother, more consistent detections.


Feature Extraction
------------------

Based on the examples in the lessons, I chose to use a combination of HOG and
color features. I felt that spatial features were redundant with HOG and also
likely to introduce overfitting to the specific images in the training set.

To extract HOG features, I used a `get_hog_features()` in `util.py`, which is
closely based on the example given in the lesson. It wraps SciKit Image's
`skimage.feature.hog` function to reduce the number of parameters required. Like
the lessons, I chose to use the `YCrCb` colorspace for this project. Because of
the way `YCrCb` splits color channels, it separates luminance from color, which
makes it good for feature detection. I chose to use all of the color channels
for HOG feature extraction.

For the other HOG parameters (orientations, pixels_per_cell, cells_per_block), I
initially tested the values used in the lesson, and I found that they worked
well (~98% accuracy), so I decided not to change them.

For color feature data, I used a 32-bin histogram on each of the three channels.
Because there are so many fewer color features than HOG features (96 vs 5292), I
don't think the color features have a large impact on detections.

For spatial feature data, I used a 32x32 bin sample of all 3 channels. Since my
HOG features and histogram are calculated form a 64x64 patch, this means that
the bins are each 2x2 pixels. Using spatial features improved my classisifier's
performance on the test set from 98% to 99%.

Here are the parameters that I ultimately used for feature extraction:

| Parameter           | Value   |
|---------------------|---------|
| Color Space         | YCrCb   |
| Histogram Bins      | 32      |
| Hog Orientations    | 9       |
| Hog Pixels Per Cell | 8       |
| Hog Cells Per Block | 2       |
| Hog Channel         | All     |
| Spatial Feature Bins| 32 x 32 |

My feature extraction methods can be found in `util.py`, but my processing
pipeline uses whole-image methods for speed, as discussed below.

Classifier
----------

For this project, I chose to use a linear Support Vector Machine (SVM). Various
background research has shown that SVM is a good compliment to HOG features, and
my results concur.

I trained my classifier from the Udacity training sets of
[vehicles](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip)
and [non-vehicles](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip)
using the included Python program  `train_classifier.py`, I was able
to consistently achieve accuracy of about 98-99% on a test set consisting of a
randomly selected 20% of the labeled data input. Before training the classifier,
the data was scaled using a `sklearn.preprocessing.StandardScaler` fit to the
training set, which scaled the data to zero mean and unit variance. To avoid
having to retrain my classifier every time I run the program, I saved the
classifier, parameters, and scaler to a Python pickle `classifier.p` to quickly
load into the detection pipeline.

Image Processing -- Sliding Window Search
-----------------------------------------

The processing pipeline is invoked from `detect_vehicles.py`, but the processing
code is in `detector.py`, which defines a `Detector` class. `Detector` is a
callable that takes in an image and returns bounding boxes of probably vehicles
using a three step pipeline:

1. At multiple scales and ROIs, perform a sliding window search using the
  feature extractor and classifier.
2. Draw the classifier "hits" (likely vehicle bounding boxes) onto a heatmap.
3. Threshold the heatmap to reduce false-positives.
4. Return bounding boxes of the thresholded heat map.

`detect_vehicles.py` then draws the bounding boxes from the `Detector` on the
image and displays or saves it as requested.

To find cars in images, I used a sliding window search at multiple scales.
Initially, I generated a set of windows and applied feature extraction to each
one independently using the methods in `util.py`, but I found that this method
was far too slow, taking tens of seconds per image. Instead, I pre-computed the
HOG image at each scale and slid the window over the HOG image for faster
processing. Similarly, I binned the image at each scale for spatial features,
although spatial feature calculation time is minimal compared to the HOG and
histogram calculation. For the histogram calculateion, I used
`skimage.filters.rank.windowed_histogram` to rapidly compute a sliding window
histogram of the whole image at multiple scales. With these methods and well-
chosen ROIs, I was able to process frames at about one every 2.4 seconds. A lot
of the processing time is spent on a single scale--the 48 pixel size. This is a
small patch that searches over a large area with a large overlap. That single
scale makes up about 1 second out of the 2.4 second processing time, but without
it, I was unable to detect the car in test image 3. For other scales, I balanced
my overlap and scales to remain accurate while keeping processing time
manageable. For small scales, I reduced the ROI and overlap to reduce the
number of windows. For larger scales I was able to use larger ROIs and more
overlap. The other scales I processed each took about 0.3 seconds per frame.

Ultimately, I decided on the following set of scales, overlaps, and ROIs:

| Window Size | Overlap | ROI (x)    | ROI (y)    |
|-------------|---------|------------|------------|
| 32 x 32     | 0.0     | [320, 960) | [396, 460) |
| 48 x 48     | 0.5     | [0, 1280)  | [360, 540) |
| 64 x 64 *   | 0.5     | [426, 853) | [396, 648) |
| 112 x 112   | 0.75    | [0, 1280)  | [360, 630) |
| 128 x 128   | 0.75    | [0, 1280)  | [360, 630) |

\* This is the native size of the feature extractor

I took three steps to improve the reliability of the classifier. The first was
that instead of using the prediction of the linear SVM, I used the distance of
the feature vector from the decision boundary as a measure of "car-ness".
Measurements with a distance of less than 0.3 were discarded as being too close
to the borderline. This greatly reduced the false positives without an
appreciable loss of true positive classifications. Next, I combined the detected
windows into a heatmap. Initially, I simply incremented pixels inside the
detection window, as demonstrated in the lesson. However, I found that this
method overweights large vehicles, because they tend to have redundant
detections at multiple scales. To reduce this, I weighted the detection windows
in the heatmap by a factor inversely proportional to the area of the window.
This way a large detection window has less influence on the heat map per pixel
than a small detection window. I used a simple threshold of 2.25 to
differentiate between "car" and "not-car" pixels, then used
`scipy.ndimage.measurements.label` to group continuous sets of "car" pixels and
drew bounding boxes around each. Finally, I rejected any bounding boxes smaller
than 32x32 pixels, which could sometimes form as a result of partially-
overlapping false positives. Using these methods, I was able to eliminate
false positives on the test images, although some detection bounding boxes
ended up being smaller than the cars they represented.

Video Processing
----------------

The output of the pipeline on the test video can be found [here](test_video_out.mp4),
and the output of the pipeline on the project video can be found
[here](project_video_out.mp4).

The main difference between the video pipeline and the single-image pipeline is
the persistence of heatmaps. Instead of basing detections on a single frame's
heatmap, the heatmaps are filtered together using an exponential filter with a
smoothing factor of 0.25. This causes heat from previous frames to persist
from frame to frame with exponential decay such that each frame's heatmap is
weighted sum of 3/4 the previous frame's heatmap plus 1/4 the current frame's
detections. This filters out jitter, but it also increases the accuracy of
detections. Because cars cannot instantaneously jump from frame to frame, the
heat from the previous frames provides more information that may not exist in
the current frame due to lighting or other factors.

Discussion
----------

The two most serious issues I faced with this task are the actual shape of the
output bounding boxes and the slowness of the search.

Many of the bounding boxes do not fully enclose the car they represent. It may
be possible to improve the bounding boxes by reducing the heatmap threshold,
but doing so often reintroduces false positives. It would be more desirable to
have a classifier with fewer false positives and to rely less on heatmaps for
reliability. I did not have time to explore other classifiers (or other
features, such as Haar-like wavelet features) to see if any are more accurate
at identifying cars. Current research suggests that a deep convolutional neural
network approach would be the most effective, but this has the downside of
requiring a GPU to operate at reasonable speed.

Currently, the speed of this method is unacceptably slow. As noted above, it can
only process one frame every 2.4 seconds. In order to operate in real time on
a vehicle, it would have to process a frame at least every 100 ms, or 24 times
as fast as the current algorithm. Reimplementing this algorithm in C++ would
probably yield an appreciable speed-up, and it may be possible to reduce the
window size from 64 x 64 to 32 x 32, which would probably speed up the HOG
and histogram calculate considerably. For further improvement, a GPU or FPGA
implementation might be necessary. It may even be possible to achieve the
desired speed using Tensor Flow.

Overall accuracy does not seem to be a major problem for this network. Changes
in lighting and shadow, which are often vexing for image processing, don't seem
to have a large effect. However, it's notable that there are no colored cars
in any of the test images or video, and those might have very different
feature responses.

Results
-------

Here are the results of running this processing pipeline on the test images:

![Test Image 1](output_images/test1.jpg)

In test image `test1.jpg`, both cars are identified, although the bounding box
on the white car is slightly small. Cars in the distance and behind the divider
are not identified.

![Test Image 2](output_images/test2.jpg)

In test image `test2.jpg`, there are no cars to identify. The car far off into
the distance is too far away.

![Test Image 3](output_images/test3.jpg)

In test image `test3.jpg`, the white car is identified, but the bounding box is
a little small.

![Test Image 4](output_images/test4.jpg)

In test image `test4.jpg`, both cars are identified. Cars in the distance and
behind the divider are not identified.

![Test Image 5](output_images/test5.jpg)

In test image `test5.jpg`, both cars are identified, although the bounding box
on the white car is slightly small. Cars in the distance and behind the divider
are not identified.

![Test Image 6](output_images/test6.jpg)

In test image `test6.jpg`, both cars are identified, although the bounding box
on the white car is slightly small. Cars in the distance and behind the divider
are not identified.
