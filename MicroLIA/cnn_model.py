

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv1D, Dense, Dropout, TimeDistributed, Flatten
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib
import os
from tqdm import tqdm

class CNNClassifier:
    """
    Time-step based 1D-CNN classifier for multi-channel light curves.
    NOTE: This model is memory-intensive and intended for use on machines with significant RAM.
    """
    def __init__(self, sequence_length=1500, num_channels=6, confidence_threshold=0.8):
        self.sequence_length = sequence_length
        self.num_channels = num_channels
        self.confidence_threshold = confidence_threshold
        self.model = None; self.history = None; self.classes_ = None; self.label_map = None
        self.scaler = StandardScaler()

    def build_model(self):
        self.model = Sequential([
            Conv1D(filters=128, kernel_size=5, activation="relu", padding="same", input_shape=(self.sequence_length, self.num_channels)),
            Dropout(0.3),
            Conv1D(filters=64, kernel_size=3, activation="relu", padding="same"),
            Dropout(0.3),
            Conv1D(filters=32, kernel_size=3, activation="relu", padding="same"),
            Dropout(0.3),
            TimeDistributed(Flatten()),
            TimeDistributed(Dense(len(self.classes_), activation="softmax"))
        ])
        self.model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    def preprocess(self, X_data, fit_scaler=False):
        nsamples, nx, ny = X_data.shape
        X_data_reshaped_2d = X_data.reshape((nsamples * nx, ny))
        if fit_scaler: self.scaler.fit(X_data_reshaped_2d)
        scaled_data = self.scaler.transform(X_data_reshaped_2d)
        return scaled_data.reshape(nsamples, nx, ny)

    def fit(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=16):
        self.classes_ = np.unique(y_train)
        self.label_map = {label: i for i, label in enumerate(self.classes_)}
        y_train_int = np.array([self.label_map[label] for label in y_train])
        y_val_int = np.array([self.label_map[label] for label in y_val])
        X_train_prep = self.preprocess(X_train, fit_scaler=True)
        X_val_prep = self.preprocess(X_val, fit_scaler=False)
        y_train_cat = tf.keras.utils.to_categorical(y_train_int, num_classes=len(self.classes_))
        y_val_cat = tf.keras.utils.to_categorical(y_val_int, num_classes=len(self.classes_))
        y_train_repeated = np.repeat(y_train_cat[:, np.newaxis, :], self.sequence_length, axis=1)
        y_val_repeated = np.repeat(y_val_cat[:, np.newaxis, :], self.sequence_length, axis=1)
        self.build_model()
        self.model.summary()
        early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
        self.history = self.model.fit(
            X_train_prep, y_train_repeated, epochs=epochs, batch_size=batch_size,
            validation_data=(X_val_prep, y_val_repeated), callbacks=[early_stop], verbose=1)

    def predict(self, X_test, threshold=None):
        if self.model is None: raise RuntimeError("Model not trained/loaded.")
        if threshold is None: threshold = self.confidence_threshold
        X_test_prep = self.preprocess(X_test, fit_scaler=False)
        predictions = self.model.predict(X_test_prep, verbose=0)
        decisions_list = []
        for sample_pred in tqdm(predictions, desc="Making Decisions"):
            decision_made = False
            for t_pred in sample_pred:
                pred_idx = np.argmax(t_pred)
                if t_pred[pred_idx] >= threshold:
                    decisions_list.append(self.classes_[pred_idx]); decision_made = True; break
            if not decision_made: decisions_list.append(self.classes_[np.argmax(sample_pred[-1])])
        return np.array(decisions_list)

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        print("\n--- Classification Report ---"); print(classification_report(y_test, y_pred, labels=self.classes_, target_names=[str(c) for c in self.classes_]))
        cm = confusion_matrix(y_test, y_pred, labels=self.classes_)
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
               xticklabels=self.classes_, yticklabels=self.classes_,
               title='Confusion Matrix', ylabel='True label', xlabel='Predicted label')
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'), ha="center", va="center", color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout(); plt.show()
    
    # ... include other methods like plot_history, get_decision_steps etc.