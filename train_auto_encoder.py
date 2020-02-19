from model import create_auto_encoder
from model import create_encoder
from model import create_decoder

auto_encoder, input_img, encoded, decoded = create_encoder()

encoder = create_encoder(input_img, encoded)

decoder = create_decoder(auto_encoder)






