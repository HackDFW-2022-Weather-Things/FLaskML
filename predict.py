import numpy as np
def predict(img, model):
    print(img.shape)
    result = model.predict(np.expand_dims(img, 0))
    print(result)
    return result