%pip install matplotlib
DATASET_PATH = ''#fill in you base path
DATASET_TRAIN = DATASET_PATH + ''#fill in you training endpoint
DATASET_TEST = DATASET_PATH + ''# fill in your testing endpoint
IMG_SIZE = (128,128) #sizeing all images to 150x150 pixel
BATCH_SIZE = 32
EPOCHS = 20

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


training_validation_data_gen=ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    brightness_range=[0.8,1.2],
    zoom_range=0.2,
    validation_split=0.2
)

testing_data_gen=ImageDataGenerator(
    rescale=1./255
)

SEED = 42  # For reproducibility
# Load train (70%) set
train_ds=training_validation_data_gen.flow_from_directory(
    DATASET_TRAIN,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training',
    seed=SEED,
    shuffle=True
)

val_ds=training_validation_data_gen.flow_from_directory(
    DATASET_TRAIN,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation',
    seed=SEED,
    shuffle=True
)

test_ds=testing_data_gen.flow_from_directory(
    DATASET_TEST,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    seed=SEED,
    shuffle=True
)

model = Sequential([
    # Convolutional Layer 1
    Conv2D(32, (3,3), activation='relu', input_shape=(128, 128, 3), name='conv_1'),
    MaxPooling2D(pool_size=(2,2)),

    # Convolutional Layer 2
    Conv2D(64, (3,3), activation='relu', name='conv_2'),
    MaxPooling2D(pool_size=(2,2)),

    # Convolutional Layer 3
    Conv2D(128, (3,3), activation='relu', name='conv_3'),
    MaxPooling2D(pool_size=(2,2)),

    Conv2D(256, (3,3), activation='relu', name='conv_4'),
    MaxPooling2D(pool_size=(2,2)),

    # Flatten the output
    Flatten(),

    # Fully Connected Layer
    Dense(256, activation='relu'),

    # Output Layer (2 classes)
    Dense(1, activation='sigmoid') 
])

model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('CatDogCNN.h5', monitor='val_loss', save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)

EPOCHS = 50

history = model.fit(
  train_ds,
  validation_data = val_ds,
  epochs = EPOCHS,
  callbacks=[early_stopping, model_checkpoint, reduce_lr]
)

# Evaluate on Test Set
test_loss, test_acc = model.evaluate(test_ds)
print(f"Test Accuracy: {test_acc:.4f}")

# Function to visualize filters
def visualize_filters(model, layer_name, num_filters=8):
    # Get the weights of the specified layer
    layer = model.get_layer(layer_name)
    filters, biases = layer.get_weights()  # Shape: (filter_height, filter_width, input_channels, num_filters)
    print(f"Filters shape for {layer_name}: {filters.shape}")

    # Normalize filter values to [0, 1] for visualization
    filters = (filters - filters.min()) / (filters.max() - filters.min())

    # Plot the filters
    fig, axes = plt.subplots(1, num_filters, figsize=(20, 2))
    for i in range(num_filters):
        if i < filters.shape[-1]:  # Ensure we don't exceed available filters
            # Extract the i-th filter (3x3x3 for RGB input in conv1)
            f = filters[:, :, :, i]
            # If RGB, average across channels or pick one channel (e.g., 0 for red)
            f = np.mean(f, axis=2) if f.shape[2] == 3 else f
            axes[i].imshow(f, cmap='gray')
            axes[i].axis('off')
            axes[i].set_title(f'Filter {i}')
    plt.suptitle(f'Filters from {layer_name}')
    plt.show()

# Visualize filters from the first convolutional layer
visualize_filters(model, 'conv_1', num_filters=8)

# Optionally visualize filters from other layers
# visualize_filters(model, 'conv2', num_filters=8)
# visualize_filters(model, 'conv3', num_filters=8)
from tensorflow.keras.preprocessing import image

# Load the image
img_path = ''  # Replace with the actual path
img = image.load_img(img_path, target_size=(128, 128))  # Resize to match your model's input

# Convert image to a NumPy array
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

# Preprocess the image (if necessary)
img_array = img_array / 255.0  # Normalize pixel values to [0, 1]

prediction = model.predict(img_array)# Make the prediction

# Interpret the prediction
if prediction[0][0] > 0.5:
    print("Prediction: Dog")
else:
    print("Prediction: Cat")