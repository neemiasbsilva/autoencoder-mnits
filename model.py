# build the simplest possible auto enconder
from keras.layers import Input, Dense
from keras.models import Model

class AutoEncode:

    def __init__(self, encoding_dim=32, shape=(784,)):
        self.enconding_dim = encoding_dim
        self.shape = shape

    