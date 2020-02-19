from model import create_auto_encoder
from model import create_encoder
from model import create_decoder
from keras.optimizers import RMSprop
from dataset_generator import datasetGenerator
import numpy as np


auto_encoder, input_img, encoded, decoded = create_auto_encoder()

encoder = create_encoder(input_img, encoded)

decoder = create_decoder(auto_encoder)

auto_encoder.compile(optimizer=RMSprop, loss='binary_cross_entropy')

dataset_generator = datasetGenerator()

x_train, x_test = dataset_generator.x_train, dataset_generator.x_test

# normalize all values between 0 and 1 and flatten
x_train = np.float32(x_train) / 255
x_test = np.float32(x_test) / 255
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

print(x_train.shape)
print(x_test.shape)


