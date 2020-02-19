from keras.models import load_model
from model import *
from dataset_generator import datasetGenerator
import numpy as np
import matplotlib.pyplot as plt

model = load_model('model_final.h5')

auto_encoder, input_img, encoded, decoded = create_auto_encoder()

dg = datasetGenerator()

x_test = dg.x_test
x_test = np.float32(x_test) / 255

x_test = x_test.reshape((len(x_test), x_test.shape[1], x_test.shape[2], 1))

decoded_imgs = auto_encoder.predict(x_test)

n = 10

plt.figure(figsize=(20, 4))

for i in range(n):
    # display original
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    plt.axis('off')

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28), cmap='gray')
    plt.axis('off')

plt.show()
