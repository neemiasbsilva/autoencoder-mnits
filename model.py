# build the simplest possible auto enconder
from keras.layers import Input, Dense
from keras.models import Model


def create_auto_encoder(encoding_dim=32, shape=(784,)):
    # Create the input placeholder, encoded representation, and lossy reconstruction

    input_img = Input(shape=shape)

    encoded = Dense(encoding_dim, activation='relu')(input_img)

    decoded = Dense(shape, activation='sigmoid')(encoded)

    autoencoder = Model(input_img, decoded)

    return autoencoder, input_img, encoded, decoded


def create_encoder(input_img, encoded):
    encoder = Model(input_img, encoded)

    return encoder


def create_decoder(auto_encoder_model, encoding_dim=32):
    encoded_input = Input(shape=(encoding_dim))

    decoder_layer = auto_encoder_model.layer[-1]

    decoder = Model(encoded_input, decoder_layer(encoded_input))

    return decoder
