######################
# Model construction #
######################

from theano import tensor

from blocks.bricks import Rectifier, MLP, Softmax
from blocks.bricks.cost import MisclassificationRate
from blocks.bricks.conv import ConvolutionalLayer, ConvolutionalSequence
from blocks.bricks.conv import MaxPooling, ConvolutionalActivation, Flattener
from blocks.initialization import IsotropicGaussian, Constant, Uniform


import logging
logging.basicConfig()


x = tensor.tensor4('features')
y = tensor.lmatrix('targets')

# Convolutional layers

conv_layers = [ConvolutionalActivation(Rectifier().apply, (3, 3), 32),
               ConvolutionalActivation(Rectifier().apply, (3, 3), 32),
               MaxPooling((2, 2)), 
               ConvolutionalActivation(Rectifier().apply, (3, 3), 64),
               ConvolutionalActivation(Rectifier().apply, (3, 3), 64),
               MaxPooling((2, 2)), 
               ConvolutionalActivation(Rectifier().apply, (3, 3), 128),
               ConvolutionalActivation(Rectifier().apply, (3, 3), 128),
               MaxPooling((2, 2)),
               ConvolutionalActivation(Rectifier().apply, (3, 3), 256),
               ConvolutionalActivation(Rectifier().apply, (3, 3), 256),
               MaxPooling((2, 2)),
               ConvolutionalActivation(Rectifier().apply, (3, 3), 512),
               ConvolutionalActivation(Rectifier().apply, (3, 3), 512),
               MaxPooling((2, 2)),
               ConvolutionalActivation(Rectifier().apply, (3, 3), 512),
               ConvolutionalActivation(Rectifier().apply, (3, 3), 512)]
convnet = ConvolutionalSequence(conv_layers, num_channels=3,
                                image_size=(260, 260),
                                weights_init=Uniform(0, 0.2),
                                biases_init=Constant(-0.01))
convnet.initialize()
print convnet.get_dim('output')
# Fully connected layers

features = Flattener().apply(convnet.apply(x))

mlp = MLP(activations=[Rectifier(), None],
          dims=[512, 256, 2], weights_init=Uniform(0, 0.2),
          biases_init=Constant(0.))
mlp.initialize()
y_hat = mlp.apply(features)

# Numerically stable softmax
#cost = Softmax().categorical_cross_entropy(y, y_hat)
#cost.name = 'nll'
y = y.flatten()
misclass = MisclassificationRate().apply(y, y_hat)
misclass.name = 'error_rate'

cost = Softmax().categorical_cross_entropy(y, y_hat)
# z = y_hat - y_hat.max(axis=1).dimshuffle(0, 'x')
# log_prob = z - tensor.log(tensor.exp(z).sum(axis=1).dimshuffle(0, 'x'))
# flat_log_prob = log_prob.flatten()
# range_ = tensor.arange(y.shape[0])
# flat_indices = y.flatten() + range_ * 2
# log_prob_of = flat_log_prob[flat_indices].reshape(y.shape, ndim=2)
# cost = -log_prob_of.mean()
cost.name = 'nll'

# Print sizes to check
print("Representation sizes:")
for layer in convnet.layers:
    print(layer.get_dim('input_'))
    print(layer.get_dim('output'))

############
# Training #
############

from blocks.main_loop import MainLoop
from blocks.graph import ComputationGraph
from blocks.extensions import FinishAfter
from blocks.algorithms import GradientDescent, Momentum
from blocks.extensions import Printing, SimpleExtension
from blocks.extensions.plot import Plot
from blocks.extensions.saveload import SerializeMainLoop
from blocks.extensions.monitoring import DataStreamMonitoring
from blocks.extensions.monitoring import TrainingDataMonitoring

from dataset import DogsVsCats
from streams import RandomPatch
from extensions import MyLearningRateSchedule, EarlyStoppingDump
from fuel.streams import DataStream
from fuel.schemes import SequentialScheme, ShuffledScheme

batch_size = 64
training_stream = DataStream(DogsVsCats('train'),
  iteration_scheme=ShuffledScheme(20000, batch_size))
training_stream = RandomPatch(training_stream, 270, (260, 260))

valid_stream = DataStream(DogsVsCats('valid'),
                iteration_scheme=SequentialScheme(2000, batch_size))
valid_stream = RandomPatch(valid_stream, 270, (260, 260))

cg = ComputationGraph([cost])
algorithm = GradientDescent(cost=cost, params=cg.parameters, step_rule=Momentum(learning_rate=1e-4,
                                                          momentum=0.9))

main_loop = MainLoop(
    data_stream=training_stream, algorithm=algorithm,
    extensions=[
        FinishAfter(after_n_epochs=200),
        TrainingDataMonitoring(
            [cost, misclass],
            prefix='train',
            after_every_epoch=True),
        DataStreamMonitoring(
            [cost, misclass],
            valid_stream,
            prefix='valid'),
        SerializeMainLoop('dogs_vs_cats.pkl', after_every_epoch=True),
        EarlyStoppingDump('/home/user/Documents/ift6266', 'valid_nll'),
        MyLearningRateSchedule('valid_nll', 2),
        Printing()
    ]
)
main_loop.run()
