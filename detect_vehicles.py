import argparse
import pickle
from detector import Detector
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip
from util import read_image, draw_boxes, rgb, write_image, convert_video_frame, make_heatmap

parser = argparse.ArgumentParser()
parser.add_argument('input_file', type=str, help='Image or video file to process')
parser.add_argument('output_file', type=str, help='Output file with boxes drawn', nargs='?')
parser.add_argument('--heat-threshold', type=float, help='Heatmap value required to activate', default=2.25)
parser.add_argument('--smoothing', type=float, help='Alpha value for heatmap smoothing filter', default=1.0)
parser.add_argument('--subclip', type=float, nargs=2, required=False, help='Beginning and end times of video')
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

if file_extension in ['jpg', 'png']:
    detector = Detector(classifier, feature_parameters, window_shape, scaler, args.heat_threshold)

    print('Loading ' + args.input_file + ' as a ' + feature_parameters['cspace'] + ' image')
    img = read_image(args.input_file, feature_parameters['cspace'])
    output_to_file = args.output_file and len(args.output_file)

    print('Detecting vehicles')
    boxes = detector(img, show_plots=(not output_to_file))
    print(boxes)
    output = draw_boxes(rgb(img, feature_parameters['cspace']), boxes)

    if output_to_file:
        print('Writing output to ' + args.output_file)
        write_image(args.output_file, output, 'RGB')
    else:
        plt.figure()
        plt.title(args.input_file)
        plt.imshow(output)
        plt.show()

elif file_extension in ['mp4']:
    detector = Detector(classifier, feature_parameters, window_shape, scaler, args.heat_threshold, alpha=args.smoothing)

    def frame_handler(frame):
        boxes = detector(convert_video_frame(frame, feature_parameters['cspace']))
        output = draw_boxes(frame, boxes)
        return output

    clip = VideoFileClip(args.input_file)

    if args.subclip and len(args.subclip) == 2:
        clip = clip.subclip(args.subclip[0], args.subclip[1])
    clip = clip.fl_image(frame_handler)

    print("Writing video file to {}".format(args.output_file))
    clip.write_videofile(args.output_file, audio=False)
else:
    raise Exception('Unidentified file extension' + file_extension)
