from keras.datasets import mnist

class datasetGenerator:

    def __init__(self):
        # we discarding the labels ( since we're only interested in encoding/decoding the inpup images)
        (x_train, _), (x_test, _) = mnist.load_data()
        self.x_train = x_train
        self.x_test = x_test
