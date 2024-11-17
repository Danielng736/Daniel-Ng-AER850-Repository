import os
import json
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight

folder_path = 'C:\\Users\\Danie\\OneDrive\\Desktop\\AER 815 Codes\\Daniel-Ng-AER850-Repository\\Project 2\\Project 2 Data\\Data' 
train_dir = os.path.join(folder_path, 'train')
validation_dir = os.path.join(folder_path, 'valid')

# Train and Validate augmentation data 
train_data = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

validation_data = ImageDataGenerator(rescale=1./255)

# Data generators
train_generator = train_data.flow_from_directory(train_dir, target_size=(500, 500), batch_size=32, class_mode='categorical')

validation_generator = validation_data.flow_from_directory(validation_dir, target_size=(500, 500), batch_size=32, class_mode='categorical')

# Calculates class weights which are used to handle imbalanced datasets
wgts = compute_class_weight('balanced', classes=np.unique(train_generator.classes), y=train_generator.classes)
wgts_dict = dict(enumerate(wgts))

# Builds the model
model = Sequential([
    Conv2D(16, (3, 3), input_shape=(500, 500, 3), activation='relu'),
    BatchNormalization(),
    LeakyReLU(alpha=0.01),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(32, (3, 3), activation='relu'),
    BatchNormalization(),
    LeakyReLU(alpha=0.01),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='softmax')
])

# Compiles the model
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])

# Early stopping
stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Training the model
hist = model.fit(train_generator, epochs=30, validation_data=validation_generator, callbacks=[stop], class_weight=wgts_dict)

model.save('model.h5')

# Evaluating the model on validation set
val_loss, val_accuracy = model.evaluate(validation_generator)
print(f'Validation Loss: {val_loss}')
print(f'Validation Accuracy: {val_accuracy}')

# Plotting training and validation loss
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(hist.history['loss'], label='Training Loss')
plt.plot(hist.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plotting training and validation accuracy
plt.subplot(1, 2, 2)
plt.plot(hist.history['accuracy'], label='Training Accuracy')
plt.plot(hist.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()
class_indices = train_generator.class_indices
with open('C:\\Users\\Danie\\OneDrive\\Desktop\\AER 815 Codes\\Daniel-Ng-AER850-Repository\\Project 2\\class_indices.json', 'w') as class_file:
    json.dump(class_indices, class_file)