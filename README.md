# Celebrity_detection
## Overview

This image classification model is developed for the identification of five celebrities: Lionel Messi, Maria Sharapova, Roger Federer, Serena Williams, and Virat Kohli. The model employs Convolutional Neural Networks (CNNs) for feature extraction and classification. The model is trained to recognize and classify celebrities based on a dataset containing images of various celebrities.  The README provides a comprehensive overview of the model architecture, training pipeline, evaluation metrics, and conclusion.

## Model Architecture

- *Input Shape*: (128, 128, 3) - 128x128 pixels RGB images
- *Convolutional Layers*:
    -MaxPooling2D with pool size (2, 2).
- *Flatten Layer*: Flattens the output for Dense layers.
- *Dense Layers*:
    - Dense (256 units) → ReLU activation
    - Dropout (dropout rate: 0.5)
    - Dense (512 units) → ReLU activation
    - Dense (5 units) → Softmax activation for multi-class classification

The model is trained using a training pipeline that includes data augmentation to increase the diversity of the training dataset. The ImageDataGenerator is employed to perform operations such as rotation, width and height shifts, shear, zoom, and horizontal flips. The dataset is split into training and testing sets, normalized to the range [0, 1], and one-hot encoded for categorical labels. Early stopping is implemented with a patience of 5 epochs, monitoring the training accuracy.

## Training Parameters

The model is trained for 200 epochs with a batch size of 128. Training involves data normalization and includes a validation split of 10%. The training history, comprising accuracy and loss, is visualized to monitor model performance.

## Model Evaluation

Model evaluation is performed on a separate test set, providing accuracy as the primary metric. Additionally, sample predictions are generated on test images to qualitatively assess the model's performance.


## Conclusion

The model achieves good accuracy of 96% on the testing set, showing it's ability to classify celebrity images. Early stopping helps prevent overfitting, and data augmentation contributes to the model's ability to generalize well to unseen data. The classification report provides detailed insights into the model's performance for each celebrity class.

## Usage

The trained model is saved as "celebrity_model.h5" for future use. To make predictions on new images, load the model and utilize the provided make_prediction function.

## Dependencies

- Python
- TensorFlow
- NumPy
- OpenCV
- scikit-learn
