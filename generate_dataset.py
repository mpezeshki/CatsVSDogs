# credits to Bart van Merrienboer
import numpy
from os import listdir
from os.path import isfile, join

import h5py
import numpy
from scipy import misc

rng = numpy.random.RandomState(54321)

read_path = '/data/lisa/data/dogs_vs_cats/train'
save_path = '/data/lisatmp3/pezeshki'
if __name__ == "__main__":
    files = [f for f in listdir(read_path)
             if isfile(join(read_path, f))]

    # Shuffle examples around
    rng.shuffle(files)

    # Create HDF5 file
    f = h5py.File(join(save_path, 'dogs_vs_cats.hdf5'), 'w')
    dt = h5py.special_dtype(vlen=numpy.dtype('uint8'))
    features = f.create_dataset('images', (25000,), dtype=dt)
    shapes = f.create_dataset('shapes', (25000, 3), dtype='uint16')
    targets = f.create_dataset('labels', (25000,), dtype='uint8')

    for i, f in enumerate(files):
        image = misc.imread(join(read_path, f))
        target = 0 if 'cat' in f else 1
        features[i] = image.flatten()
        targets[i] = target
        shapes[i] = image.shape
        if i % 200 == 0:
            print 'iteration: ' + str(i) + '  shape: ' + str(image.shape)
        print '{:.0%}\r'.format(i / 25000.),