from keras.models import load_model
from model import *
from dataset_generator import datasetGenerator

model = load_model('model_final.h5')

auto_encoder, input_img, encoded, decoded = create_auto_encoder()

encoder = create_encoder(input_img, encoded)

decoder = create_decoder(auto_encoder)

dg = datasetGenerator()

x_test = dg.x_test

encoded_imgs = encoder.predict(x_test)

decoded_imgs = decoder.predict(encoded_imgs)

