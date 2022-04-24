import os
import numpy as np
import pickle as p

n_images = 9
im_sqrt = int(np.sqrt(n_images))
label = 8

data = p.load(open(os.path.join('data', 'test.p'), 'rb'))

images = []
for _ in range(n_images):
    idx = 0
    while len(images) < n_images:
        if data['y'][idx] == label:
            images.append(data['X'][idx])
            idx += 1
        else:
            idx += 1

