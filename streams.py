from collections import OrderedDict

import theano
import numpy
from scipy.misc import imresize

from fuel.transformers import Transformer


class RandomPatch(Transformer):
    def __init__(self, data_stream, scale_size, patch_size, source='features'):
        super(RandomPatch, self).__init__(data_stream)
        self.scale_size = scale_size
        self.patch_size = patch_size
        self.patch_source = source

    def get_data(self, request=None):
        if request is not None:
            raise ValueError
        patch_height, patch_width = self.patch_size
        data = OrderedDict(zip(self.sources, next(self.child_epoch_iterator)))
        new_data = OrderedDict(zip(self.sources, [[] for _ in self.sources]))
        for i, image in enumerate(data[self.patch_source]):
            # Resize the image
            image_height, image_width, _ = image.shape
            scaling = float(self.scale_size) / min(image_height, image_width)
            image = imresize(image, scaling)
            # Select random patch
            image_height, image_width, _ = image.shape
            x = image_width - patch_width
            y = image_height - patch_height
            if x:
                x = numpy.random.randint(x)
            if y:
                y = numpy.random.randint(y)
            patch = image[y:y + patch_width, x:x + patch_height]
            # Convert to float and c, 0, 1 format
            patch = (patch.transpose((2, 0, 1)).astype(theano.config.floatX) /
                     255)
            new_data[self.patch_source].append(patch)
            for source in self.sources:
                if source != self.patch_source:
                    new_data[source].append(data[source][i])
        return tuple(numpy.asarray(source_data)
                     for source_data in new_data.values())
