
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv1D, Dense, Dropout, GlobalMaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib
import os

class CNNClassifier:
    def __init__(self, sequence_length=1200):
        self.sequence_length = sequence_length
        self.model = None; self.history = None; self.classes_ = None
        self.scaler = StandardScaler()

    def build_model(self, features_per_step=1, num_classes=2):
        model = Sequential()
        model.add(Conv1D(filters=128, kernel_size=5, activation="relu", padding="same", input_shape=(self.sequence_length, features_per_step)))
        model.add(Dropout(0.3)); model.add(Conv1D(filters=64, kernel_size=3, activation="relu", padding="same"))
        model.add(Dropout(0.3)); model.add(Conv1D(filters=32, kernel_size=3, activation="relu", padding="same"))
        model.add(GlobalMaxPooling1D()); model.add(Dense(num_classes, activation="softmax"))
        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]); return model

    def _prepare_data(self, X_data, fit_scaler=False):
        if fit_scaler: self.scaler.fit(X_data)
        scaled_data = self.scaler.transform(X_data)
        return scaled_data.reshape(-1, self.sequence_length, 1)

    def fit(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=64):
        self.classes_ = np.unique(y_train)
        num_classes = len(self.classes_)
        label_map = {label: i for i, label in enumerate(self.classes_)}
        y_train_int = np.array([label_map[label] for label in y_train])
        y_val_int = np.array([label_map[label] for label in y_val])
        X_train_prep = self._prepare_data(X_train, fit_scaler=True)
        X_val_prep = self._prepare_data(X_val, fit_scaler=False)
        y_train_cat = tf.keras.utils.to_categorical(y_train_int, num_classes=num_classes)
        y_val_cat = tf.keras.utils.to_categorical(y_val_int, num_classes=num_classes)
        self.model = self.build_model(features_per_step=1, num_classes=num_classes)
        early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
        self.history = self.model.fit(
            X_train_prep, y_train_cat, epochs=epochs, batch_size=batch_size,
            validation_data=(X_val_prep, y_val_cat), callbacks=[early_stop], verbose=1)

    def predict(self, X_test):
        if self.model is None: raise RuntimeError("Model has not been trained or loaded.")
        X_test_prep = self._prepare_data(X_test, fit_scaler=False)
        predictions = self.model.predict(X_test_prep)
        predicted_indices = np.argmax(predictions, axis=1)
        return self.classes_[predicted_indices]

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        print("\n--- Classification Report ---"); print(classification_report(y_test, y_pred, target_names=[str(c) for c in self.classes_]))
        cm = confusion_matrix(y_test, y_pred, labels=self.classes_)
        plt.figure(figsize=(8, 6)); sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[str(c) for c in self.classes_], yticklabels=[str(c) for c in self.classes_])
        plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Confusion Matrix"); plt.show()
    
    def plot_history(self):
        if not self.history: return
        plt.figure(figsize=(12, 5)); plt.subplot(1, 2, 1)
        plt.plot(self.history.history["loss"], label="Train Loss"); plt.plot(self.history.history["val_loss"], label="Validation Loss")
        plt.title("Model Loss"); plt.xlabel("Epoch"); plt.legend()
        plt.subplot(1, 2, 2); plt.plot(self.history.history["accuracy"], label="Train Accuracy"); plt.plot(self.history.history["val_accuracy"], label="Validation Accuracy")
        plt.title("Model Accuracy"); plt.xlabel("Epoch"); plt.legend(); plt.tight_layout(); plt.show()