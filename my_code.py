import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16, VGG19, ResNet50

# Prepare the data
train_path = "data/Train_Validation sets"
test_path = "data/Independent Test Set"

# Data augmentation
datagen = ImageDataGenerator(
    rescale=1/255.0,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
train_generator = datagen.flow_from_directory(train_path, target_size=(64, 64), color_mode='rgb', class_mode='sparse', batch_size=64)
test_generator = datagen.flow_from_directory(test_path, target_size=(64, 64), color_mode='rgb', class_mode='sparse', batch_size=64, shuffle=False)

# Define custom CNN model
def create_custom_model():
    model = Sequential([
        Conv2D(64, (3,3), input_shape=(64, 64, 3), padding="same"),
        LeakyReLU(),
        Conv2D(64, (3,3), padding="same"),
        LeakyReLU(),
        BatchNormalization(),
        MaxPooling2D((2,2)),
        Dropout(0.3),

        Conv2D(128, (3,3), padding="same"),
        LeakyReLU(),
        Conv2D(128, (3,3), padding="same"),
        LeakyReLU(),
        BatchNormalization(),
        MaxPooling2D((2,2)),
        Dropout(0.4),

        Conv2D(256, (3,3), padding="same"),
        LeakyReLU(),
        Conv2D(256, (3,3), padding="same"),
        LeakyReLU(),
        Conv2D(256, (3,3), padding="same"),
        LeakyReLU(),
        BatchNormalization(),
        MaxPooling2D((2,2)),
        Dropout(0.5),

        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(4, activation="softmax")
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Define VGG16 model with fine-tuning
def create_vgg16_model():
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(64, 64, 3))
    for layer in base_model.layers[:-4]:  # Freeze all layers except the last 4
        layer.trainable = False
    x = base_model.output
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(4, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Define VGG19 model with fine-tuning
def create_vgg19_model():
    base_model = VGG19(weights='imagenet', include_top=False, input_shape=(64, 64, 3))
    for layer in base_model.layers[:-4]:  # Freeze all layers except the last 4
        layer.trainable = False
    x = base_model.output
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(4, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Define ResNet50 model with fine-tuning
def create_resnet50_model():
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(64, 64, 3))
    for layer in base_model.layers[:-4]:  # Freeze all layers except the last 4
        layer.trainable = False
    x = base_model.output
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(4, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Create callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

# Train and evaluate models
models = [create_custom_model(), create_vgg16_model(), create_vgg19_model(), create_resnet50_model()]
model_names = ['Custom Model', 'VGG16', 'VGG19', 'ResNet50']
accuracies = []

for model, name in zip(models, model_names):
    print(f"Training {name}...")
    history = model.fit(train_generator, validation_data=test_generator, epochs=30, callbacks=[early_stopping, reduce_lr], verbose=1)
    loss, accuracy = model.evaluate(test_generator)
    accuracies.append(accuracy)
    print(f"{name} Accuracy: {accuracy * 100:.2f}%")

# Visualize the comparison
plt.figure(figsize=(10, 6))
plt.bar(model_names, accuracies, color=['blue', 'green', 'red', 'purple'])
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Comparison of Model Accuracies')
plt.ylim(0, 1)
plt.show()
