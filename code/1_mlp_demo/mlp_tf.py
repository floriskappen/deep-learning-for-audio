import numpy as np
from random import random
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Array([[0.1, 0.2], [0.2, 0.2]])
# Array([[0.3], [0.4]])
def generate_dataset(num_samples, test_size):
  # Create a dataset to train a network for the sum operation
  x = np.array([[random() / 2 for _ in range(2)] for _ in range(num_samples)]) # Array([[0.1, 0.2], [0.3, 0.4]])
  y = np.array([[i[0] + i[1]] for i in x]) # Array([[0.3], [0.7]])
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
  return x_train, x_test, y_train, y_test

if __name__ == "__main__":
  x_train, x_test, y_train, y_test = generate_dataset(5000, 0.3)
  print(f"x_test: \n {x_test}")
  print(f"y_test: \n {y_test}")

  # Build model:  2 -> 5 -> 1
  model = tf.keras.Sequential([
    tf.keras.layers.Dense(5, input_dim=2, activation="sigmoid"),
    tf.keras.layers.Dense(1, activation="sigmoid"),
  ])

# Compile model
optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)
model.compile(optimizer=optimizer, loss="MSE")

# Train model
model.fit(x_train, y_train, epochs=100)

# Evaluate model
print("\nModel evalutation")
model.evaluate(x_test, y_test, verbose=1)

# Make predictions
data = np.array([[0.1, 0.2], [0.2, 0.2]])
predictions = model.predict(data)

print("\nSome predictions:")
for d, p in zip(data, predictions):
  print(f"{d[0]} + {d[0]} = {p[0]}")
