from sklearn.model_selection import train_test_split
import numpy as np
from MicroLIA import create_cnn_set, CNNClassifier, noise_models
import random

def main():
    timestamps = [
        np.linspace(0, 1000, 1500), np.sort(np.random.uniform(0, 1000, 800)),
        np.concatenate([np.linspace(0, 200, 400), np.linspace(800, 1000, 400)])
    ]
    sample_baselines = np.array([18, 19, 20, 21, 22])
    sample_rms = np.array([0.01, 0.02, 0.04, 0.08, 0.15])
    noise_model = noise_models.create_noise(sample_baselines, sample_rms)
    light_curves, labels = create_cnn_set(
        timestamps=timestamps, n_events_per_class=200, n_points=1500, noise_model=noise_model)
    X_train, X_test, y_train, y_test = train_test_split(
        light_curves, labels, test_size=0.3, random_state=42, stratify=labels)
    X_train_sub, X_val, y_train_sub, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)
    cnn = CNNClassifier(sequence_length=1500)
    cnn.fit(X_train_sub, y_train_sub, X_val, y_val, epochs=50, batch_size=32)
    cnn.evaluate(X_test, y_test)
    cnn.plot_history()

if __name__ == '__main__':
    main()