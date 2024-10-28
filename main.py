from random import random
from numpy import load
from numpy import zeros
from numpy import ones
from numpy import asarray
from numpy.random import randint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Model
#from tensorflow.keras.models import Input
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Concatenate
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from matplotlib import pyplot
from models.generator import resnet_block, define_generator
from models.discriminator import define_discriminator
from models.composite_model import define_composite_model
from util import load_real_samples, generate_real_samples, generate_fake_samples, save_models, summarize_performance, update_image_pool

# load image data
dataset = load_real_samples('horse2zebra_256.npz')
print('Loaded', dataset[0].shape, dataset[1].shape)

# define input shape based on the loaded dataset
image_shape = dataset[0].shape[1:]
print(image_shape)

# generator: A -> B
g_model_AtoB = define_generator(image_shape)
g_model_AtoB.summary()

# generator: B -> A
g_model_BtoA = define_generator(image_shape)
g_model_BtoA.summary()

# discriminator: A -> [real/fake]
d_model_A = define_discriminator(image_shape)
d_model_A.summary()

# discriminator: B -> [real/fake]
d_model_B = define_discriminator(image_shape)
d_model_B.summary()

# composite: A -> B -> [real/fake, A]
c_model_AtoB = define_composite_model(g_model_AtoB, d_model_B, g_model_BtoA, image_shape)
c_model_AtoB.summary()

# composite: B -> A -> [real/fake, B]
c_model_BtoA = define_composite_model(g_model_BtoA, d_model_A, g_model_AtoB, image_shape)
c_model_BtoA.summary()

# train models
train(d_model_A, d_model_B, g_model_AtoB, g_model_BtoA, c_model_AtoB, c_model_BtoA, dataset)