# Region Proposal Network (RPN)
The folder contains a RPN implementation for generating regions from Basal Cell images to estimate cell density.

### Prerequisites
Running **train/predict** requires correct path to the input data and the following **packages** for ```python-3.x```

```
matplotlib==3.1.1
opencv-python==4.2.0
pandas==1.4.0
numpy==1.17.4
scikit-image==0.15.0
torch==1.4.0+cu92
torchvision==0.5.0+cu92
```
### Annotation Fromat

```image_name, cell_type, xmin, ymin, xmax, ymax  ```
* **example**: 001.jpg, 'CELL', 277, 9, 381, 109

#### Running Training Script
```python train.py -i "path_to_images\\" -a "path_to_annotation_file (train.csv)"```

#### Running Evaluation Script
```python evaluate.py -i "path_to_images\\" -a "path_to_annotation_file (val.csv)" -f "path_to_weights (checkpoints\\RPN_CELL.pth)"```

#### Running Prediction Scripts
```python predict.py -i "sample_img.jpg" -f "path_to_weights (checkpoints\\RPN_CELL.pth)"```

### Data Statistics

* Total Images in dataset = 40
* Number of Images for training = 30
* Number of Images for validation = 10

### Data Processing
* The Basal images are normalized to a mean of and a variance of 0.5. The pixel values are scaled between 0 and 1.
* For the purpose of using CNNs with CUDA, the data was resized to a tensor size of [1, 400, 400].

### Model Parameter

* **Loss function**: Smooth-L1
* **Optimizer**: Adam
* **Epoch**: 200
* **Learning Rate**: 0.005 (reduces 1/10 if validation loss does not increase for 5 epochs) 

### Results
#### Control Cases
![Alt text](sample_prediction/control_case.png?raw=true "Sample Predictions")

#### Mild Cases
![Alt text](sample_prediction/mild_case.png?raw=true "Sample Predictions")

#### Moderate Cases
![Alt text](sample_prediction/moderate_case.png?raw=true "Sample Predictions")

#### Severe Cases
![Alt text](sample_prediction/severe_case.png?raw=true "Sample Predictions")

##### The model shows reasonable performance on Mild Cases, however, further training is required and evaluation to be done before any concrete conclusions are drawn.
