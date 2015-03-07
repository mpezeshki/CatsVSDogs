# credits to Bart van Merrienboer
import numpy
from os import listdir
from os.path import isfile, join

import h5py
import numpy
from scipy import misc

rng = numpy.random.RandomState(123522)

path = '/home/user/Data/cats_vs_dogs'
if __name__ == "__main__":
    files = [f for f in listdir(path)
             if isfile(join(path, f))]

    # Shuffle examples around
    rng.shuffle(files)

    # Create HDF5 file
    f = h5py.File(join(path, 'dogs_vs_cats.hdf5'), 'w')
    dt = h5py.special_dtype(vlen=numpy.dtype('uint8'))
    features = f.create_dataset('images', (25000,), dtype=dt)
    shapes = f.create_dataset('shapes', (25000, 3), dtype='uint16')
    targets = f.create_dataset('labels', (25000,), dtype='uint8')

    for i, f in enumerate(files):
        image = misc.imread(join(path, f))
        target = 0 if 'cat' in f else 1
        features[i] = image.flatten()
        targets[i] = target
        shapes[i] = image.shape
        print image.shape
        print '{:.0%}\r'.format(i / 25000.),