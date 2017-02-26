
class Detector(object):
    def __init__(self, classifier, feature_parameters):
        self._classifier = classifier
        self._feature_parameters = feature_parameters
    def __call__(self, img):
        raise Exception("Detector not yet implemented")
