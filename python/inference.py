from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.preprocessing import image
import sys
import numpy as np

if len(sys.argv) != 3:
    print("Usage: python3 inference.py <model.h5> <image>")
    sys.exit(0)

f = sys.argv[2]

net = load_model(sys.argv[1])


print(f)
img = image.load_img(f, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis = 0)
pred = net.predict(x)[0]
top_inds = pred.argsort()[::-1][:5]
for i in top_inds:
    print('{}:    {:.3f}'.format(int(i), pred[i]))
