# Model Architecture

inputs = keras.Input(shape=(window_size, features))
x = layers.LSTM(512,return_sequences=True)(inputs)
x = layers.Dropout(0.5)(x)
x = layers.LSTM(128,return_sequences=True)(x)
outputs = layers.Dense(12,activation='softmax')(x)

mdl = keras.Model(inputs=inputs, outputs=outputs, name='HAR_model')

mdl.compile(loss=tf.keras.losses.categorical_crossentropy,
            optimizer='RMSProp',
            metrics=['accuracy'])

print(mdl.summary())
epochs = 20

history = mdl.fit(train_ds, epochs=epochs,
                  validation_data=validation_ds)