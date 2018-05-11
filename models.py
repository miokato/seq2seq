from keras.layers import Input, Dense, LSTM, RepeatVector
from keras.models import Model
from keras.layers.core import Flatten
from keras.optimizers import SGD, RMSprop, Adam
from keras.layers.wrappers import Bidirectional
from keras.layers.wrappers import TimeDistributed
from keras.layers import multiply
from keras.layers.normalization import BatchNormalization


def model():
    timesteps = 50
    inputs = Input(shape=(timesteps, 128))
    encoded = LSTM(512)(inputs)
    inputs_a = inputs
    inputs_a = Dense(2048)(inputs_a)
    inputs_a = BatchNormalization()(inputs_a)
    a_vector = Dense(512, activation='softmax')(Flatten()(inputs_a))
    mul = multiply([encoded, a_vector])
    encoder = Model(inputs, mul)

    v = RepeatVector(timesteps)(mul)
    v = Bidirectional(LSTM(512, return_sequences=True))(v)
    decoded = TimeDistributed(Dense(128, activation='softmax'))(v)

    autoencoder = Model(inputs, decoded)
    autoencoder.compile(optimizer=Adam(), loss='categorical_crossentropy')

    return encoder, autoencoder


def model2():
    timesteps = 50
    inputs = Input(shape=(timesteps, 128))
    encoded = LSTM(512)(inputs)

    inputs_a = Input(shape=(timesteps, 128))
    a_vector = Dense(512, activation='softmax')(Flatten()(inputs))
    # mul         = merge([encoded, a_vector],  mode='mul')  # this for keras v1
    mul = multiply([encoded, a_vector])
    encoder = Model(inputs, mul)

    x = RepeatVector(timesteps)(mul)
    x = Bidirectional(LSTM(512, return_sequences=True))(x)
    decoded = TimeDistributed(Dense(128, activation='softmax'))(x)

    autoencoder = Model(inputs, decoded)
    autoencoder.compile(optimizer=Adam(), loss='categorical_crossentropy')

    return encoder, autoencoder
