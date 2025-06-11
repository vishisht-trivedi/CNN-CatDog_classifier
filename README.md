# Cat vs Dog Image Classifier 🐱🐶

This project is a Convolutional Neural Network (CNN) based image classifier built with TensorFlow/Keras. It is trained to distinguish between images of cats and dogs using a dataset from Kaggle. The code is contained in a Jupyter Notebook: `CatDogCNN.ipynb`.

## 🧠 Model Overview

- Built using TensorFlow and Keras
- Convolutional layers for feature extraction
- Trained on the standard Cats vs Dogs dataset from Kaggle
- Validation and accuracy tracking

## 🛠️ Requirements

- Python 3.7+
- Google Colab or Jupyter Notebook
- TensorFlow
- NumPy
- Matplotlib

## 📂 Usage in Google Colab

1. Open the notebook in Google Colab.

2. **Mount Google Drive**:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

3. **Upload Test Image to Google Drive**:
   - Upload your test image (e.g. `test.jpg`) to a known folder in your Google Drive, e.g. `/content/drive/MyDrive/CatDogTest/test.jpg`.

4. **Run Prediction on Custom Image**:
   Add the following code block at the end of the notebook to test your image:
   ```python
   import cv2
   import numpy as np
   from tensorflow.keras.preprocessing.image import load_img, img_to_array

   # Load image
   img_path = '/content/drive/MyDrive/CatDogTest/test.jpg'
   img = load_img(img_path, target_size=(150, 150))
   img_array = img_to_array(img) / 255.0
   img_array = np.expand_dims(img_array, axis=0)

   # Predict
   prediction = model.predict(img_array)
   if prediction[0] > 0.5:
       print("It's a Dog 🐶")
   else:
       print("It's a Cat 🐱")
   ```

## 📊 Output

The notebook includes visualizations for:

- Model accuracy & loss over epochs
- Example predictions
- Image preprocessing pipeline

## 🧪 Testing

You can test the model using custom images after training. Just ensure the image size is 150x150 or it will be resized before prediction.

---

## 📎 Dataset

You can get the dataset from [Kaggle Cats vs Dogs Dataset](https://www.kaggle.com/c/dogs-vs-cats/data) and upload it to your Google Drive for use with this notebook.

---

## 📄 License

MIT License - feel free to use and modify.
