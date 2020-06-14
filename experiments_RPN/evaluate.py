from model_class import RPN
import matplotlib.pyplot as plt
import argparse
import numpy as np
from skimage import io
from skimage.color import rgb2gray
import cv2
import os
import pandas as pd
from os import listdir
from os.path import isfile, join

def get_args():
    parser = argparse.ArgumentParser(description='Prediction from RPN Network', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-n', '--num_classes', metavar='N', type=int, default=2,
                        help='Number of Classes')
    parser.add_argument('-i', '--path_to_images', metavar='PI', type=str, default=None,
                        help='Path to Images')
    parser.add_argument('-a', '--path_to_annotation', metavar='PA', type=str, default=None,
                        help='Path to Annotatins')
    parser.add_argument('-f', '--load', dest='load', type=str, default=None,
                        help='Load model from a .pth file')
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    if args.path_to_images and args.path_to_annotation and args.load:
        coord_file = pd.read_csv(args.path_to_annotation)
        model_obj = RPN(path_to_dict=args.load, num_classes=args.num_classes)

        img_files = [os.path.join(args.path_to_images, f) for f in listdir(args.path_to_images) if isfile(join(args.path_to_images, f))]
        for index, arg in enumerate(img_files):
            file_name = arg.split("\\")[-1]
            print("Evaluating:", file_name)
            bbox, scores = model_obj.get_prediction(arg)
            sorted_scores = sorted(scores, reverse=True)
            if len(bbox) > 0:
                if len(np.unique(bbox)) > 1:
                    if len(bbox) > 5:
                        bbox = bbox[:5]

                    image = rgb2gray(io.imread(arg))
                    img_rgb = cv2.cvtColor(image.astype('float32'), cv2.COLOR_GRAY2RGB)
                    # Load GT
                    name_filter = coord_file["image_names"] == file_name
                    filtered = coord_file[name_filter]
                    xmin, ymin, xmax, ymax = filtered.iloc[0]['xmin'], filtered.iloc[0]['ymin'], filtered.iloc[0][
                        'xmax'], filtered.iloc[0]['ymax']
                    GT = [xmin, ymin, xmax, ymax]

                    # Draw ALL BOXES
                    for arg in bbox:
                        iou = "{:.2f}".format(model_obj._bb_intersection_over_union(GT, arg))
                        cv2.rectangle(img_rgb, (arg[0], arg[1]), (arg[2], arg[3]), color=(0, 0, 255), thickness=2)
                        cv2.putText(img_rgb, iou, (arg[0], arg[1]), cv2.FONT_HERSHEY_SIMPLEX,
                                            1, (255, 255, 255), 2, cv2.LINE_AA)

                    cv2.rectangle(img_rgb, (GT[0], GT[1]), (GT[2], GT[3]), color=(0, 255, 0), thickness=2)

                    plt.imshow(img_rgb)
                    plt.show()
                else:
                    print("NO Bounding Box Detected!")
            else:
                print("No Bounding Box Detected for {}".fromat(file_name))
                continue
    else:
        print("IMAGE PATH AND MODEL WEIGHTS ARE REQUIRED!")


