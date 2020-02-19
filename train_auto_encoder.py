from model import create_auto_encoder
from model import create_encoder
from model import create_decoder
from dataset_generator import datasetGenerator
from keras.optimizers import Adadelta


import numpy as np


auto_encoder, input_img, encoded, decoded = create_auto_encoder()

encoder = create_encoder(input_img, encoded)

decoder = create_decoder(auto_encoder)
learning_rate = 0.01
momentum = 0.9
decay = 0.0005

# optimizer = SGD(lr=learning_rate, momentum=momentum, decay=decay, nesterov=False)

optimizer = Adadelta(learning_rate=0.01)

auto_encoder.compile(optimizer=optimizer, loss='binary_crossentropy')

dataset_generator = datasetGenerator()

x_train, x_test = dataset_generator.x_train, dataset_generator.x_test

# normalize all values between 0 and 1 and flatten
x_train = np.float32(x_train) / 255
x_test = np.float32(x_test) / 255
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

# train autoencoder for 50 epochs
nb_epochs = 50
auto_encoder.fit(x_train, x_train, epochs=nb_epochs, shuffle=True, validation_data=(x_test, x_test))

# save the model
auto_encoder.save("model_final.h5")
