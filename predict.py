import glob

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

labels = {
    0: "airplane",
    1: "automobile",
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",
    7: "horse",
    8: "ship",
    9: "truck",
}


def predict():
    for filename in glob.iglob("testImages/*"):
        # read the image into memory
        foo = Image.open(filename)
        foo = foo.resize((32, 32), Image.ANTIALIAS)
        foo.save("resized.jpeg")
        x = imageio.imread("resized.jpeg")
        print("Image " + filename)
        # make it the right size
        x = x.reshape(-1, 32, 32, 3)
        # in our computation graph
        # perform the prediction
        out = model.predict(x)
        for i in range(out.shape[1]):
            print(labels[i] + " " + str(out[0][i]))


predict()
