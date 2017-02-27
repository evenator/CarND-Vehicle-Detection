import argparse
import pickle
from detector import Detector
import matplotlib.pyplot as plt
from util import read_image, draw_boxes, rgb, write_image

parser = argparse.ArgumentParser()
parser.add_argument('input_file', type=str, help='Image or video file to process')
parser.add_argument('output_file', type=str, help='Output file with boxes drawn', nargs='?')
parser.add_argument('--heat-threshold', type=float, help='Heatmap value required to activate', default=1.0)
args = parser.parse_args()

print('Loading classifier from pickle classifier.p')
with open('classifier.p', 'rb') as f:
    data = pickle.load(f)
    classifier = data['classifier']
    feature_parameters = data['feature_parameters']
    window_shape = data['shape']
    scaler = data['scaler']
print('Feature parameters:')
print(feature_parameters)
file_extension = args.input_file.split('.')[-1].lower()

detector = Detector(classifier, feature_parameters, window_shape, scaler, args.heat_threshold)

if file_extension in ['jpg', 'png']:
    print('Loading ' + args.input_file + ' as a ' + feature_parameters['cspace'] + ' image')
    img = read_image(args.input_file, feature_parameters['cspace'])
    print('Detecting vehicles')
    boxes = detector(img)
    output = draw_boxes(rgb(img, feature_parameters['cspace']), boxes)
    if args.output_file and not args.output_file.empty():
        print('Writing output to ' + args.output_file)
        write_image(args.output_file, output, 'RGB')
    else:
        plt.figure()
        plt.title(args.input_file)
        plt.imshow(output)
        plt.show()
elif file_extension in ['mp4']:
    raise Exception('Video processing not yet implemented')
else:
    raise Exception('Unidentified file extension' + file_extension)
