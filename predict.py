import numpy as np
def predict(img, model):
    model.predict(np.expand_dims(img, 0))
    return 