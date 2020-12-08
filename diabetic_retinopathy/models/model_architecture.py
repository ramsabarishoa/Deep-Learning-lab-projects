num_classes = 2

model = Sequential([
  layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256,256,3)),
  layers.MaxPooling2D(pool_size=(2, 2)),
  layers.Conv2D(64, (3, 3), activation='relu'),
  layers.MaxPooling2D(pool_size=(2, 2)),
  layers.Conv2D(128, (3, 3), activation='relu'),
  layers.MaxPooling2D(pool_size=(2, 2)),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dropout(0.2),
  layers.Dense(64,activation='relu'),
  layers.Dropout(0.2),
  layers.Dense(num_classes)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

print(model.summary())

oldStdout = sys.stdout
log_file = open('logFile', 'w')
sys.stdout = log_file

epochs=20
history = model.fit(
  train_dataset,
  validation_data=val_dataset,
  epochs=epochs
)

sys.stdout = oldStdout