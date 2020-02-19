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






