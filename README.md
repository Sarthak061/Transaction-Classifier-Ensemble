# Transaction Classification using Deep Learning Ensemble

## Overview
This project implements an ensemble deep learning model for classifying transactions into "Yes" or "No" categories based on image data. The ensemble combines MobileNetV2, ResNet50, and VGG16 models to improve classification accuracy. The trained model is exported in ONNX format for further deployment.

The directory of this project is as follows:
```
Transaction-Classifier-Ensemble
├── README.md
├── requirement.txt
└── Transaction_Classifier_ensemble.ipynb (Jupyter Notebook - training)
```

## Dataset
The dataset used for training can be downloaded as follows:
```sh
!pip install -q gdown  

!gdown --fuzzy "https://drive.google.com/file/d/1NBZn0kuutPfimytF6Rw-fIVxNTSjrYTU/view"
```

It consists of images categorized into "Yes" and "No" classes, making it efficiently a Binary Classification Task. 

The dataset is split into training, validation, and test sets in the following proportions:
- **Training**: 60%
- **Validation**: 30%
- **Test**: 10%

### Steps to Load the Dataset
1. **Mount Google Drive (if using Colab):**
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
2. **Extract dataset from ZIP file in Drive:**
   ```python
   import zipfile
   with zipfile.ZipFile('/content/drive/MyDrive/dataset.zip', 'r') as zip_ref:
       zip_ref.extractall('/content/')
   ```
3. **Alternatively, manually upload the dataset to Colab:**
   - Click on the **Files** tab in Colab.
   - Upload the dataset ZIP file.
   - Use `!unzip dataset.zip -d /content/` to extract the dataset.

4. **The easiest approach - Use gdown to download:**

    ```sh
    !pip install -q gdown  

    !gdown --fuzzy "https://drive.google.com/file/d/1NBZn0kuutPfimytF6Rw-fIVxNTSjrYTU/view"

    !unzip transaction-yes-no.zip -d destination_folder

    ```


## Dependencies
To run this project, clone this repository:
```
!gitclone https://github.com/Sarthak061/Transaction-Classifier-Ensemble
```

Next install the necessary dependencies:
```sh
!pip install -r requirements.txt 

```

## Setup Environment
1. Install required dependencies using the command mentioned above.
2. Mount Google Drive and extract the dataset if using Google Colab.

## Dataset Preprocessing
1. The dataset is extracted from a ZIP file located in Google Drive or manually uploaded.
2. The `split_dataset` function organizes the dataset into training, validation, and test directories.
3. Data augmentation is applied to the training dataset to enhance generalization.
4. Class imbalance is handled using class weighting during model training.

## Training and Testing the Model
2. Training involves preprocessing the data, training three models: MobileNetV2, ResNet50, and VGG16.
3. Their outputs are averaged to form an ensemble model, which helps stabilize predictions and reduce bias.
4. The trained model is evaluated on the test dataset to compute metrics including accuracy, precision, recall, and AUC.

## Model Architecture
Three models are used in this approach:
- **MobileNetV2**
- **ResNet50**
- **VGG16**

Each model is independently trained, and their outputs are averaged to form an ensemble model. The ensemble approach helps in stabilizing predictions and reducing bias.

## Model Export
The ensemble model is exported using TensorFlow SavedModel format and then converted to ONNX format using `tf2onnx`:
```sh
python -m tf2onnx.convert --saved-model ensemble_model_saved --output ensemble_model.onnx
```

## Loading and Testing the ONNX Model
To load and test the ONNX model within the Jupyter Notebook, use the following Python code:
```python
import onnxruntime as ort
import numpy as np
import cv2

def load_onnx_model(model_path):
    session = ort.InferenceSession(model_path)
    return session

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    image = image.astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=0)  
    return image

def predict_onnx(session, input_data):
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    return session.run([output_name], {input_name: input_data})

onnx_model = load_onnx_model("ensemble_model.onnx")

image_path = "path/to/your/image.jpg"  # Replace with actual image path
input_image = preprocess_image(image_path)
prediction = predict_onnx(onnx_model, input_image)
print("Prediction:", prediction)
```

## Results
The model's performance was evaluated using the following metrics on a held-out test set:

| Metric        | Score         |
| ------------- |:-------------:|
| Accuracy      | 0.9395        |
| Precision     | 0.8447        |
| Recall        | 0.9645        |
| F1-score      | 0.9007        |
| AUC           | 0.9909        |


These results demonstrate strong performance across all evaluated metrics, indicating that the ensemble model is effective at classifying transactions.  The high AUC score further suggests excellent discriminatory power.

## Usage
1. Install dependencies.
2. Mount Google Drive and extract the dataset or manually upload it to Colab.
3. Run the notebook to preprocess data, train models, and generate results.
4. Export the trained model for further deployment.
5. Load and test the ONNX model for inference using an image.
