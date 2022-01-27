import typing as t
import tensorflow as tf
from tensorflow.keras import layers, models


def Activation(activation, **kwargs):
  if activation in ['lrelu', 'leakyrelu']:
    return layers.LeakyReLU(**kwargs)
  else:
    return layers.Activation(activation, **kwargs)


def Normalization(normalizer, **kwargs):
  if normalizer in ['layer_norm', 'layernorm']:
    return layers.LayerNormalization(**kwargs)
  elif normalizer in ['batch_norm', 'batchnorm']:
    return layers.BatchNormalization(**kwargs)
  elif normalizer in ['instance_norm', 'instancenorm']:
    return tfa.layers.InstanceNormalization(**kwargs)
  raise NameError(f'Unknown normalization: {normalizer}')


def channel_attention(inputs,
                      in_channel,
                      reduction_ratio: int = 8,
                      name: str = 'ChannelAttention'):
  reshape = layers.Reshape((1, 1, in_channel), name=f'{name}/reshape')
  dense1 = layers.Dense(in_channel // reduction_ratio,
                        kernel_initializer='he_normal',
                        use_bias=True,
                        bias_initializer='zeros',
                        name=f'{name}/dense1')
  activation = Activation('relu', name=f'{name}/relu')
  dense2 = layers.Dense(in_channel,
                        kernel_initializer='he_normal',
                        use_bias=True,
                        bias_initializer='zeros',
                        name=f'{name}/dense2')

  avg_pool = layers.GlobalAveragePooling2D(name=f'{name}/average_pool')(inputs)
  avg_pool = reshape(avg_pool)
  avg_pool = dense1(avg_pool)
  avg_pool = activation(avg_pool)
  avg_pool = dense2(avg_pool)

  max_pool = layers.GlobalMaxPooling2D(name=f'{name}/max_pool')(inputs)
  max_pool = reshape(max_pool)
  max_pool = dense1(max_pool)
  max_pool = activation(max_pool)
  max_pool = dense2(max_pool)

  outputs = layers.Add(name=f'{name}/add')([avg_pool, max_pool])
  outputs = Activation('sigmoid', name=f'{name}/sigmoid')(outputs)
  outputs = layers.Multiply(name=f'{name}/multiply')([outputs, inputs])
  return outputs


def spatial_attention(inputs, kernel_size=7, name: str = 'SpatialAttention'):
  avg_pool = layers.Lambda(lambda x: tf.reduce_mean(x, axis=-1, keepdims=True),
                           name=f'{name}/average_pool')(inputs)
  max_pool = layers.Lambda(lambda x: tf.reduce_max(x, axis=-1, keepdims=True),
                           name=f'{name}/max_pool')(inputs)
  outputs = layers.Concatenate(axis=-1,
                               name=f'{name}/concat')([avg_pool, max_pool])
  outputs = layers.Conv2D(filters=1,
                          kernel_size=kernel_size,
                          strides=1,
                          padding='same',
                          activation='sigmoid',
                          use_bias=False,
                          kernel_initializer='he_normal',
                          name=f'{name}/conv')(outputs)
  outputs = Normalization('batchnorm', name=f'{name}/batchnorm')(outputs)
  outputs = layers.Multiply(name=f'{name}/multiply')([outputs, inputs])
  return outputs


def cbam(inputs, in_channel, name: str = 'CBAM'):
  """ 
  Convolutional Block Attention Modules
  Paper: https://arxiv.org/abs/1807.06521
  Official PyTorch implementation: https://github.com/Jongchan/attention-module
  """
  outputs = channel_attention(inputs, in_channel, name=f'{name}/channel')
  outputs = spatial_attention(outputs, name=f'{name}/spatial')
  outputs = Activation('sigmoid', name=f'{name}/activation')(outputs)
  return outputs


def residual_block(inputs,
                   filters: int = 64,
                   kernel_size: int = 3,
                   strides1: int = 1,
                   use_projection: bool = False,
                   use_cbam: bool = False,
                   normalization='batchnorm',
                   activation='relu',
                   name='residual_block'):
  shortcut = inputs
  if use_projection:
    shortcut = layers.Conv2D(filters,
                             kernel_size=1,
                             strides=strides1,
                             padding='same',
                             use_bias=False,
                             name=f'{name}/projection_conv')(shortcut)
    shortcut = Normalization(normalization,
                             name=f'{name}/projection_norm')(shortcut)

  outputs = layers.Conv2D(filters,
                          kernel_size=1,
                          strides=strides1,
                          use_bias=False,
                          name=f'{name}/conv1')(inputs)
  outputs = Normalization(normalization, name=f'{name}/normalization1')(outputs)
  outputs = Activation(activation, name=f'{name}/activation1')(outputs)

  outputs = layers.Conv2D(filters,
                          kernel_size,
                          strides=1,
                          padding='same',
                          use_bias=False,
                          name=f'{name}/conv2')(outputs)
  outputs = Normalization(normalization, name=f'{name}/normalization2')(outputs)
  if use_cbam:
    outputs = cbam(inputs=outputs, in_channel=filters, name=f'{name}/cbam')
  outputs = layers.Add(name=f'{name}/add')([outputs, shortcut])
  outputs = Activation(activation, name=f'{name}/activation2')(outputs)
  return outputs


def residual_stack(inputs,
                   filters: int,
                   strides1: int = 1,
                   num_blocks: int = 2,
                   use_cbam: bool = False,
                   name: str = 'residual_stack'):
  outputs = residual_block(inputs=inputs,
                           filters=filters,
                           strides1=strides1,
                           use_projection=True,
                           use_cbam=use_cbam,
                           name=f'{name}/block_1')
  for i in range(1, num_blocks):
    outputs = residual_block(inputs=outputs,
                             filters=filters,
                             strides1=1,
                             use_cbam=use_cbam,
                             name=f'{name}/block_{i + 1}')
  return outputs


def resnet(input_shape,
           num_classes,
           normalization='batchnorm',
           activation='relu',
           use_cbam: bool = False,
           name: str = 'ResNet'):
  inputs = tf.keras.Input(shape=input_shape, name='input')

  outputs = layers.Conv2D(64,
                          kernel_size=7,
                          strides=1,
                          padding='same',
                          use_bias=False,
                          name='input/conv')(inputs)
  outputs = Normalization(normalization, name='input/normalization')(outputs)
  outputs = Activation(activation, name='input/activation')(outputs)
  outputs = residual_stack(inputs=outputs,
                   filters=16,
                   strides1=1,
                   num_blocks=2,
                   use_cbam=use_cbam,
                   name='stack_1')
  outputs = residual_stack(inputs=outputs,
               filters=16,
               strides1=1,
               num_blocks=2,
               use_cbam=use_cbam,
               name='stack_2')
  outputs = residual_stack(inputs=outputs,
               filters=16,
               strides1=1,
               num_blocks=2,
               use_cbam=use_cbam,
               name='stack_3')
  outputs = residual_stack(inputs=outputs,
               filters=16,
               strides1=1,
               num_blocks=2,
               use_cbam=use_cbam,
               name='stack_4')
  outputs = residual_stack(inputs=outputs,
               filters=16,
               strides1=1,
               num_blocks=2,
               use_cbam=use_cbam,
               name='stack_5')
  outputs = residual_stack(inputs=outputs,
               filters=16,
               strides1=1,
               num_blocks=2,
               use_cbam=use_cbam,
               name='stack_6')
  outputs = residual_stack(inputs=outputs,
               filters=16,
               strides1=1,
               num_blocks=2,
               use_cbam=use_cbam,
               name='stack_7')
  outputs = residual_stack(inputs=outputs,
             filters=16,
             strides1=1,
             num_blocks=2,
             use_cbam=use_cbam,
             name='stack_8')
  outputs = residual_stack(inputs=outputs,
                   filters=32,
                   strides1=2,
                   num_blocks=2,
                   use_cbam=use_cbam,
                   name='stack_9')
  outputs = residual_stack(inputs=outputs,
                   filters=64,
                   strides1=2,
                   num_blocks=2,
                   use_cbam=use_cbam,
                   name='stack_10')
  outputs = residual_stack(inputs=outputs,
                   filters=128,
                   strides1=2,
                   num_blocks=2,
                   use_cbam=use_cbam,
                   name='stack_11')
  outputs = layers.GlobalAveragePooling2D(name='output/pooling')(outputs)
  outputs = layers.Dense(num_classes, name='output/dense')(outputs)

  return tf.keras.Model(inputs=inputs, outputs=outputs, name=name)
