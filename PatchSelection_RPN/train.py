from model_class import RPN
import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Train the RPN on images with annotations', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-n', '--num_class', metavar='N', type=int, default=2,
                        help='Number of Class')
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=300,
                        help='Number of epochs')
    parser.add_argument('-i', '--path_to_images', metavar='PI', type=str, default=None,
                        help='Path to Images')
    parser.add_argument('-a', '--path_to_annotate', metavar='PA', type=str, default=None,
                        help='Path to Annotation')
    parser.add_argument('-b', '--batch_size', metavar='B', type=int, nargs='?', default=1,
                        help='Batch size')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.005,
                        help='Learning rate')
    parser.add_argument('-f', '--load', dest='load', type=str, default=None,
                        help='Load model from a .pth file')
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    if args.path_to_images and args.path_to_annotate:
        model_obj = RPN(path_to_dict=args.load, num_classes=args.num_class)
        # TO START TRAINING
        model_obj.train_model(args.path_to_images, args.path_to_annotate, batch_size=args.batch_size, num_epochs=args.epochs)
    else:
        print("PATH TO IMAGES AND ANNOTATION IS REQUIRED!")
