"""
BTCS301P - Experiment 10
Character, Digit, or Face Classification using Convolutional Neural Networks

This implementation uses the MNIST handwritten digit dataset and TensorFlow/Keras.

Run:
    python cnn_mnist_classification.py

Optional:
    python cnn_mnist_classification.py --epochs 5 --batch-size 64
"""

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="CNN-based MNIST digit classification."
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    return parser.parse_args()


def build_model(learning_rate):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),

        # Convolution + activation
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),

        # Fully connected part
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.30),

        # 10 MNIST classes: digits 0-9
        tf.keras.layers.Dense(10, activation="softmax")
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def plot_training_history(history):
    # Accuracy curve
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title("CNN Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig("training_accuracy.png", dpi=150)
    plt.show()

    # Loss curve
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("CNN Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("training_loss.png", dpi=150)
    plt.show()


def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=np.arange(10)
    )

    fig, ax = plt.subplots(figsize=(8, 8))
    display.plot(ax=ax, values_format="d")
    ax.set_title("MNIST CNN Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.show()


def show_sample_predictions(model, x_test, y_test):
    predictions = model.predict(x_test[:12], verbose=0)
    predicted_labels = np.argmax(predictions, axis=1)

    plt.figure(figsize=(10, 7))

    for i in range(12):
        plt.subplot(3, 4, i + 1)
        plt.imshow(x_test[i].squeeze(), cmap="gray")
        plt.title(
            f"Actual: {y_test[i]} | Pred: {predicted_labels[i]}"
        )
        plt.axis("off")

    plt.tight_layout()
    plt.savefig("sample_predictions.png", dpi=150)
    plt.show()


def main():
    args = parse_arguments()

    print("=" * 65)
    print("EXPERIMENT 10: CNN-BASED MNIST DIGIT CLASSIFICATION")
    print("=" * 65)

    # Step 1 & 2: Load standard MNIST dataset.
    print("\n[1] Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    print(f"Training images: {x_train.shape}")
    print(f"Testing images : {x_test.shape}")

    # Step 3: Normalize images and prepare CNN input shape.
    print("\n[2] Preprocessing images...")
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    print("Pixel values normalized to [0, 1].")
    print("Image size: 28 x 28 x 1")

    # Step 4 & 5: Design and configure CNN.
    print("\n[3] Building CNN model...")
    model = build_model(args.learning_rate)
    model.summary()

    # Step 6: Train CNN.
    print("\n[4] Training CNN...")
    history = model.fit(
        x_train,
        y_train,
        validation_split=0.10,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=1
    )

    # Save model.
    model.save("mnist_cnn.keras")
    print("\n[OK] Model saved as mnist_cnn.keras")

    # Step 6: Learning curves.
    print("\n[5] Generating training/validation curves...")
    plot_training_history(history)

    # Step 7: Evaluate on test set.
    print("\n[6] Evaluating model on test dataset...")
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)

    probabilities = model.predict(x_test, verbose=0)
    y_pred = np.argmax(probabilities, axis=1)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test, y_pred, average="weighted", zero_division=0
    )
    recall = recall_score(
        y_test, y_pred, average="weighted", zero_division=0
    )
    f1 = f1_score(
        y_test, y_pred, average="weighted", zero_division=0
    )

    print("\n" + "-" * 65)
    print("EVALUATION RESULTS")
    print("-" * 65)
    print(f"Test Loss       : {test_loss:.4f}")
    print(f"Accuracy        : {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Precision       : {precision:.4f}")
    print(f"Recall          : {recall:.4f}")
    print(f"F1-Score        : {f1:.4f}")
    print("-" * 65)

    # Step 7: Confusion matrix.
    print("\n[7] Generating confusion matrix...")
    plot_confusion_matrix(y_test, y_pred)

    # Step 8: Predictions on unseen test images.
    print("\n[8] Displaying sample predictions...")
    show_sample_predictions(model, x_test, y_test)

    print("\n" + "=" * 65)
    print("EXPERIMENT COMPLETED SUCCESSFULLY")
    print("=" * 65)
    print("\nGenerated files:")
    print("  mnist_cnn.keras")
    print("  training_accuracy.png")
    print("  training_loss.png")
    print("  confusion_matrix.png")
    print("  sample_predictions.png")


if __name__ == "__main__":
    main()
