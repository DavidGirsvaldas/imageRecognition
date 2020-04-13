import imageio
import numpy as np
from PIL import Image
from keras.models import model_from_json


def init():
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded Model from disk")
    # compile and evaluate loaded model
    loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return loaded_model


model = init()


def predict():
    # read the image into memory
    foo = Image.open("test1.jpeg")
    foo = foo.resize((32, 32), Image.ANTIALIAS)
    foo.save("test4.jpeg")
    x = imageio.imread("test4.jpeg")
    print(x.shape)
    # make it the right size
    x = x.reshape(-1, 32, 32, 3)
    # in our computation graph
    # perform the prediction
    out = model.predict(x)
    # make class predictions with the model
    print(out)
    print(np.argmax(out, axis=1))
    # convert the response to a string
    response = np.argmax(out, axis=1)
    return str(response[0])


predict()
