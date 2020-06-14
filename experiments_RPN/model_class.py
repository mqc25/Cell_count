import numpy as np
import torch
import cv2
from dataloader import DataProcessor
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from skimage import io
import torchvision
from skimage.color import rgb2gray
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

class RPN:
    def __init__(self, path_to_dict=None, num_classes=2):
        self.path_to_dict = path_to_dict
        self.model = self._get_model_instance_segmentation(num_classes)
        # Check for CUDA
        train_on_gpu = torch.cuda.is_available()
        if not train_on_gpu:
            self.device = torch.device("cpu")
            print("="*30)
            print("Running on CPU")
            print("=" * 30)
        else:
            print("=" * 30)
            self.device = torch.device("cuda:0")
            print("CUDA is available!")
            print("=" * 30)
        if self.path_to_dict:
            print("WEIGHTS LOADED!")
            weights = torch.load(path_to_dict, map_location=self.device)
            self.model.load_state_dict(weights)
        # Load model on CUDA/CPU
        self.model.to(self.device)

    # HELPER FUNCTIONS
    def _get_model_instance_segmentation(self, num_classes):
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        return model

    def _get_default_transform(self):
        custom_transforms = []
        custom_transforms.append(torchvision.transforms.ToTensor())
        return torchvision.transforms.Compose(custom_transforms)

    def _bb_intersection_over_union(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = abs(max(0, xB - xA + 1) * max(0, yB - yA + 1))
        if interArea == 0:
            return 0.0
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou

    def _keep_Highscore(self, output):
        processed_dict = []
        for i in range(len(output)):
            out = output[i]['boxes'].cpu().numpy().squeeze()
            if len(out.shape) > 1:
                if len(out) == 0:
                    processed_dict.append({'boxes': []})
                else:
                    scores = output[i]['scores'].cpu().numpy().squeeze()
                    processed_dict.append({'boxes': out[np.argmax(scores)]})
            else:
                processed_dict.append({'boxes': out})
        return processed_dict

    def _calculate_ious(self, annotations, output):
        iou_over_batch = 0.0
        for i in range(len(output)):
            out = output[i]['boxes']
            annotation = annotations[i]['boxes'].cpu().numpy().squeeze()
            if len(out) == 0 or len(annotation) == 0:
                iou = 0.0
            else:
                iou = self._bb_intersection_over_union(annotation, out)
            iou_over_batch += iou
        return iou_over_batch / len(output)

    def _get_transform(self):
        custom_transforms = []
        custom_transforms.append(torchvision.transforms.ToTensor())
        return torchvision.transforms.Compose(custom_transforms)

    def _collate_fn(self, batch):
        return tuple(zip(*batch))

    # TRAIN FUNCTION
    def train_model(self, path_to_images=None, path_to_annotate=None, transformation=None, batch_size=1, learning_rate=0.005, num_epochs=300):
        if path_to_images and path_to_annotate:
            if transformation is None:
                dataset = DataProcessor(imgs_dir=path_to_images, csv_path=path_to_annotate, transformations=self._get_transform(),
                                        resize_img=400)
            else:
                dataset = DataProcessor(imgs_dir=path_to_images, csv_path=path_to_annotate, transformations=transformation, resize_img=400)
            print("Images for Training:", len(dataset))
            trainloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=self._collate_fn)
            print("=" * 30)
            params = [p for p in self.model.parameters() if p.requires_grad]
            optimizer = optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=0.0005)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

            train_loss_min = np.Inf
            for epoch in range(num_epochs):
                train_loss = 0.0
                i = 0
                epoch_loss = []
                for imgs, annotations in trainloader:
                    i += 1
                    imgs = list(img.to(self.device, dtype=torch.float) for img in imgs)
                    annotations = [{k: v.to(self.device) for k, v in t.items()} for t in annotations]
                    loss_dict = self.model(imgs, annotations)
                    losses = sum(loss for loss in loss_dict.values())
                    optimizer.zero_grad()
                    losses.backward()
                    optimizer.step()
                    epoch_loss.append(float(losses.item() * len(imgs)))
                    detached_loss = losses.detach().item() * len(imgs)
                    train_loss += detached_loss

                scheduler.step(np.mean(epoch_loss))
                train_loss = train_loss / len(trainloader)
                print("Epoch:{}/{}\t Training Loss:{:.6f}".format(epoch, num_epochs, train_loss))

                if train_loss < train_loss_min:
                    print("Training loss decreased: ({:.6f} --> {:.6f}).  Saving model ...".format(train_loss_min,
                                                                                                   train_loss))
                    print("-" * 40)
                    # Save model
                    torch.save(self.model.state_dict(), 'RPN_CELL.pth')
                    train_loss_min = train_loss

    # INFERENCE FUNCTION
    def get_prediction(self, path_to_image=None):
        if path_to_image:
            data = []
            image = rgb2gray(io.imread(path_to_image))
            dataset = DataProcessor(imgs_dir=None, csv_path=None)
            processed = dataset.preprocess(image, new_size=400, normalize=True, img_transforms=self._get_default_transform())
            data.append(processed.to(self.device, dtype=torch.float))
            with torch.no_grad():
                self.model.eval()
                output = self.model(data)

        bbox = np.squeeze(output[0]['boxes'].cpu().numpy())
        scores = output[0]['scores'].cpu().numpy()
        return bbox, scores

    def evaluate(self, path_to_images=None, path_to_annotation=None):
        if path_to_images and path_to_annotation:
            pawss

