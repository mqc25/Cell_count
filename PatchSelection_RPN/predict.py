from model_class import RPN
import matplotlib.pyplot as plt
import argparse
import numpy as np
from skimage import io
from skimage.color import rgb2gray
import cv2

def get_args():
    parser = argparse.ArgumentParser(description='Prediction from RPN Network', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-n', '--num_classes', metavar='N', type=int, default=2,
                        help='Number of Classes')
    parser.add_argument('-i', '--path_to_image', metavar='PI', type=str, default=None,
                        help='Path to Images')
    parser.add_argument('-f', '--load', dest='load', type=str, default=None,
                        help='Load model from a .pth file')
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    if args.path_to_image and args.load:
        model_obj = RPN(path_to_dict=args.load, num_classes=args.num_classes)
        bbox = model_obj.get_prediction(args.path_to_image)
        if len(bbox) > 0:
            if len(np.unique(bbox)) > 1:
                image = rgb2gray(io.imread(args.path_to_image))
                img_rgb = cv2.cvtColor(image.astype('float32'), cv2.COLOR_GRAY2RGB)
                for arg in bbox:
                    cv2.rectangle(img_rgb, (arg[0], arg[1]), (arg[2], arg[3]), color=(0, 0, 255), thickness=2)
                cv2.imwrite('prediction.png', img_rgb)
                print("PREDICTION SAVED!")
                plt.imshow(img_rgb)
                plt.show()
            else:
                print("NO Bounding Box Detected!")
        else:
            print("NO Bounding Box Detected!")
    else:
        print("IMAGE PATH AND MODEL WEIGHTS ARE REQUIRED!")


