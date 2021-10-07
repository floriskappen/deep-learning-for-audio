import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras

DATA_PATH = "data.json"

def load_data(data_path):

  with open (data_path, "r") as fp:
    data = json.load(fp)
  
  X = np.array(data["mfcc"])
  y = np.array(data["labels"])
  mapping = np.array(data["mapping"])
  return X, y, mapping


def prepare_datasets(test_size, validation_size):
  
  # Load data
  X, y, mapping = load_data(DATA_PATH)

  # Create train/test split
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

  # Create test/validation split
  X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_size)

  return X_train, X_validation, X_test, y_train, y_validation, y_test, mapping


def build_model(input_shape):

  # Create model
  model = keras.Sequential()

  # 2 LSTM layers
  model.add(keras.layers.LSTM(units=64,
                              input_shape=input_shape,
                              return_sequences=True))
  model.add(keras.layers.LSTM(units=64))

  # Dense layer
  model.add(keras.layers.Dense(units=64,
                               activation="relu"))
  model.add(keras.layers.Dropout(rate=0.3))

  # Output layer
  model.add(keras.layers.Dense(10, activation="softmax"))

  return model


def predict(model, X, y, mapping):

  X = X[np.newaxis, ...]

  # prediction = [ [0.1, 0.2, ...] ]
  prediction = model.predict(X) # X -> (1, 130, 13, 1)

  # Extract index with max value
  predicted_index = np.argmax(prediction, axis=1) # [3]
  print(f"Expected mapping: {mapping[y]}, Predicted mapping: {mapping[predicted_index]}")


if __name__ == "__main__":
  
  # Create train, validation and test sets
  X_train, X_validation, X_test, y_train, y_validation, y_test, mapping = prepare_datasets(0.25, 0.2)

  # Build the CNN net
  input_shape = (X_train.shape[1], X_train.shape[2])
  model = build_model(input_shape)

  # Compile the model
  optimizer = keras.optimizers.Adam(learning_rate=0.0001)
  model.compile(optimizer=optimizer,
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"])

  # Train the CNN
  model.fit(X_train, y_train, validation_data=(X_validation, y_validation), batch_size=32, epochs=30)

  # Evaluate the CNN on the test set
  test_error, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
  print(f"Accuracy on test set is: {test_accuracy}")

  # Make predictions on a sample
  X = X_test[100]
  y = y_test[100]

  predict(model, X, y, mapping)
