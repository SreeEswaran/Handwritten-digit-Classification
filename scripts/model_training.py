from model_building import build_model
from data_preprocessing import load_and_preprocess_data
from tensorflow.keras.optimizers import Adam

X_train, X_test, y_train, y_test = load_and_preprocess_data()
model = build_model()

model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, batch_size=128, validation_data=(X_test, y_test))
model.save('../models/mnist_cnn_model.h5')
