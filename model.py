# build the simplest possible auto enconder
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model


def create_auto_encoder(encoding_dim=32, shape=(28, 28, 1)):
    # Create the input placeholder, encoded representation, and lossy reconstruction

    input_img = Input(shape=shape)

    x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    # at this point the representation is (4, 4, 8) i.e. 128-dimensional

    x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    auto_encoder = Model(input_img, decoded)

    return auto_encoder, input_img, encoded, decoded


def create_encoder(input_img, encoded):
    encoder = Model(input_img, encoded)

    return encoder


def create_decoder(auto_encoder_model, encoding_dim=32):
    encoded_input = Input(shape=(encoding_dim,))

    decoder_layer = auto_encoder_model.layers[-1]

    decoder = Model(encoded_input, decoder_layer(encoded_input))

    return decoder
