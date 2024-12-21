import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPooling3D, GlobalAveragePooling3D, Dense, Dropout, LSTM, TimeDistributed, Input
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import cv2

# Enable GPU Growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# Dataset Path
dataset_path = r'C:\Users\Jayasairamsuji\Desktop\Mini_project-main\fo3\fo3'

# Hyperparameters
IMG_SIZE = 112  # Reduced from 224
FRAMES = 30     # Reduced from 90
BATCH_SIZE = 4  # Reduced from 8
EPOCHS = 25
LEARNING_RATE = 0.001

# Data Generator Function
def video_generator(dataset_path, batch_size, frames, img_size):
    classes = os.listdir(dataset_path)
    class_indices = {cls: idx for idx, cls in enumerate(classes)}

    while True:
        data = []
        labels = []
        for cls in classes:
            cls_path = os.path.join(dataset_path, cls)
            videos = os.listdir(cls_path)
            label = class_indices[cls]

            for video in videos:
                video_path = os.path.join(cls_path, video)
                frames_list = []
                cap = cv2.VideoCapture(video_path)
                frame_count = 0
                while cap.isOpened() and frame_count < frames:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame = cv2.resize(frame, (img_size, img_size)).astype('float32') / 255.0
                    frames_list.append(frame)
                    frame_count += 1
                cap.release()

                # Padding or truncating frames
                if len(frames_list) < frames:
                    frames_list += [np.zeros((img_size, img_size, 3), dtype=np.float32)] * (frames - len(frames_list))
                elif len(frames_list) > frames:
                    frames_list = frames_list[:frames]

                data.append(frames_list)
                labels.append(label)

                if len(data) == batch_size:
                    yield np.array(data), np.array(labels)
                    data, labels = [], []

# Model Building
def build_model():
    model = Sequential()

    # Define Input Layer
    model.add(Input(shape=(FRAMES, IMG_SIZE, IMG_SIZE, 3)))

    # 3D CNN Layers
    model.add(Conv3D(32, kernel_size=(3, 3, 3), activation='relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(Conv3D(64, kernel_size=(3, 3, 3), activation='relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))

    # Replace Flatten with GlobalAveragePooling3D
    model.add(GlobalAveragePooling3D())

    # LSTM Layer
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))

    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    return model

# Compile Model
model = build_model()

# Checkpoint to save best model
checkpoint = ModelCheckpoint(r'C:\Users\Jayasairamsuji\Desktop\Mini_project-main\best_model.keras', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)

# Generator Setup
train_gen = video_generator(dataset_path, BATCH_SIZE, FRAMES, IMG_SIZE)
steps_per_epoch = 75 // BATCH_SIZE

# Model Training
history = model.fit(train_gen, steps_per_epoch=steps_per_epoch, epochs=EPOCHS, callbacks=[checkpoint])

# # Evaluate Model
# best_model = tf.keras.models.load_model('best_model.keras')
# loss, accuracy = best_model.evaluate(train_gen, steps=steps_per_epoch)
# print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')
