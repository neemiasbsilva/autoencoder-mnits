# build the simplest possible auto enconder
from keras.layers import Input, Dense
from keras.models import Model


def create_auto_encoder(encoding_dim= 32, shape=(784,)):
    # Create the input placeholder, encoded representation, and lossy reconstruction

    input_img = Input(shape=shape)

    encoded = Dense(encoding_dim, activation='relu')(input_img)

    decoded = Dense(shape, activation='sigmoid')(encoded)

    autoencoder = Model(input_img, decoded)

    return autoencoder, input_img, encoded, decoded


def create_encoder(input_img, encoded):

    encoder = Model(input_img, encoded)

    return encoder

