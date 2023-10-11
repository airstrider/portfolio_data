# ----------------
# Goal
# val_loss: 0.14
# val_acc: 0.93
# ----------------

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint

train_dataset = tfds.load('iris', split='train[:80%]')
valid_dataset = tfds.load('iris', split='train[80%:]')


def preprocess(data):
    # Should return features and one-hot encoded labels
    x = data['features']
    y = data['label']
    y = tf.one_hot(y, 3)
    return x, y


def solution_model():
    train_data = train_dataset.map(preprocess).batch(10)
    valid_data = valid_dataset.map(preprocess).batch(10)

    model = Sequential([
        Dense(512, activation='relu', input_shape=(4,)),
        Dense(256, activation='relu'),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(3, activation=tf.nn.softmax)
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy', metrics=['acc'])

    checkpoint_path = 'portfolio_data/data/temp_checkpoint.ckpt'
    checkpoint = ModelCheckpoint(filepath=checkpoint_path,
                                 save_weights_only=True,
                                 save_best_only=True,
                                 monitor='val_loss',
                                 verbose=1)

    model.fit(train_data,
              validation_data=(valid_data),
              epochs=20,
              callbacks=[checkpoint])

    model.load_weights(checkpoint_path)
    return model


if __name__ == '__main__':
    model = solution_model()
    model.save('portfolio_data/data/tensorflow-iris.h5')
