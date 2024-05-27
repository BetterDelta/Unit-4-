import tensorflow as tf
from tensorflow.keras import layers, models, datasets
import numpy as np

def build_lenet5_classic(input_shape=(32, 32, 1)):
    model = models.Sequential([
        layers.Conv2D(6, (5, 5), activation='sigmoid', input_shape=input_shape, padding='valid'),
        layers.AveragePooling2D((2, 2), strides=2),
        layers.Conv2D(16, (5, 5), activation='sigmoid', padding='valid'),
        layers.AveragePooling2D((2, 2), strides=2),
        layers.Conv2D(120, (5, 5), activation='sigmoid'),
        layers.Flatten(),
        layers.Dense(84, activation='sigmoid'),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def build_lenet5_optimized(input_shape=(32, 32, 1)):
    model = models.Sequential([
        layers.Conv2D(6, (5, 5), activation='relu', input_shape=input_shape, padding='valid', kernel_initializer='he_uniform'),
        layers.AveragePooling2D((2, 2), strides=2),
        layers.BatchNormalization(),
        layers.Conv2D(16, (5, 5), activation='relu', padding='valid', kernel_initializer='he_uniform'),
        layers.AveragePooling2D((2, 2), strides=2),
        layers.BatchNormalization(),
        layers.Conv2D(120, (5, 5), activation='relu', kernel_initializer='he_uniform'),
        layers.BatchNormalization(),
        layers.Flatten(),
        layers.Dense(84, activation='relu', kernel_initializer='he_uniform'),
        layers.BatchNormalization(),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def preprocess_images_labels(images, labels):
    images = np.pad(images, ((0, 0), (2, 2), (2, 2)), mode='constant', constant_values=0)
    images = images[..., np.newaxis] / 255.0
    labels = tf.keras.utils.to_categorical(labels, 10)
    return images, labels

def main():
    # Load and preprocess data
    (train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
    train_images, train_labels = preprocess_images_labels(train_images, train_labels)
    test_images, test_labels = preprocess_images_labels(test_images, test_labels)

    # Train classic model
    classic_model = build_lenet5_classic()
    classic_model.summary()
    classic_model.fit(train_images, train_labels, epochs=10, validation_split=0.1)
    test_loss, test_acc = classic_model.evaluate(test_images, test_labels)
    print(f"Classic Model Test Accuracy: {test_acc * 100:.2f}%")

    # Train optimized model
    optimized_model = build_lenet5_optimized()
    optimized_model.summary()
    optimized_model.fit(train_images, train_labels, epochs=10, validation_split=0.1)
    test_loss, test_acc = optimized_model.evaluate(test_images, test_labels)
    print(f"Optimized Model Test Accuracy: {test_acc * 100:.2f}%")

if __name__ == '__main__':
    main()

