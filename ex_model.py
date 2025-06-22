from tensorflow.keras import layers, models, Input

sequence_input = Input(shape=(10, 224, 224, 3))

# CNN Backbone (pre-trained, tidak di-train ulang)
cnn_base = tf.keras.applications.ResNet50(include_top=False, pooling='avg', input_shape=(224, 224, 3))
cnn_base.trainable = False  # freeze dulu untuk transfer learning

# CNN untuk tiap frame (pakai TimeDistributed)
x = layers.TimeDistributed(cnn_base)(sequence_input)  # (None, 10, 2048)

# LSTM dengan Dropout
x = layers.LSTM(128, return_sequences=True)(x)
x = layers.Dropout(0.3)(x)
x = layers.LSTM(128)(x)
x = layers.Dropout(0.3)(x)

# Fully Connected
x = layers.Dense(64, activation='relu')(x)
x = layers.Dropout(0.3)(x)

output = layers.Dense(6, activation='softmax')(x)

model = models.Model(inputs=sequence_input, outputs=output)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
