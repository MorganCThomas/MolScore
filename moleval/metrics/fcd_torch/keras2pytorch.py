import argparse
import sys
import tensorflow as tf
from torch import nn
from keras.backend.tensorflow_backend import set_session
import keras.backend as K
from keras.models import load_model
import torch


def masked_accuracy(y_true, y_pred):
    mask_value = 0.5
    a = K.sum(K.cast(K.equal(y_true, K.round(y_pred)), K.floatx()))
    c = K.sum(K.cast(K.not_equal(y_true, mask_value), K.floatx()))
    acc = (a) / c
    return acc


def build_masked_loss(loss_function, mask_value):
    """Builds a loss function that masks based on targets

    Args:
        loss_function: The loss function to mask
        mask_value: The value to mask in the targets

    Returns:
        function: a loss function that acts like loss_function
        with masked inputs.
    """

    def masked_loss_function(y_true, y_pred):
        mask = K.cast(K.not_equal(y_true, mask_value), K.floatx())
        return loss_function(y_true * mask, y_pred * mask)

    return masked_loss_function


def load_keras_model(model_file):
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.device('/cpu'):
        sess = tf.Session(config=config)
        set_session(sess)
        K.clear_session()
        masked_loss_function = build_masked_loss(K.binary_crossentropy, 0.5)
        model = load_model(
            model_file,
            custom_objects={
                'masked_loss_function': masked_loss_function,
                'masked_accuracy': masked_accuracy
            }
        )
        model.pop()
        model.pop()
    return model


def convert_LSTM_layer(keras_layer):
    pytorch_config = {
        'input_size': keras_layer.input.shape[-1].value,
        'hidden_size': keras_layer.output.shape[-1].value,
        'batch_first': True
    }
    pytorch_layer = nn.LSTM(**pytorch_config)
    W_ih, W_hh, b_ih = keras_layer.get_weights()
    pytorch_state = pytorch_layer.state_dict()
    for weight in pytorch_state:
        if weight.startswith('weight_ih'):
            pytorch_state[weight] = torch.tensor(W_ih.T)
        elif weight.startswith('weight_hh'):
            pytorch_state[weight] = torch.tensor(W_hh.T)
        elif weight.startswith('bias_hh'):
            pytorch_state[weight] *= 0
        elif weight.startswith('bias_ih'):
            pytorch_state[weight] = torch.tensor(b_ih)
        else:
            raise ValueError("Unknown parameter {}".format(weight))
    pytorch_layer.load_state_dict(pytorch_state)
    other_info = {
        'reverse': keras_layer.get_config()['go_backwards'],
        'last': not keras_layer.get_config()['return_sequences']
    }
    return pytorch_state, pytorch_config, other_info


def convert_Conv1d_layer(keras_layer):
    keras_config = keras_layer.get_config()
    pytorch_config = {
        'in_channels': keras_layer.input.shape[-1].value,
        'out_channels': keras_layer.output.shape[-1].value,
        'kernel_size': keras_config['kernel_size'][0],
        'stride': keras_config['strides'][0],
        'padding': 0,
        'dilation': keras_config['dilation_rate'],
        'bias': keras_config['use_bias'],
    }
    kernel = keras_layer.get_weights()

    pytorch_layer = nn.Conv1d(**pytorch_config)
    pytorch_state = pytorch_layer.state_dict()
    if len(kernel) > 1:
        kernel, bias = kernel
    else:
        kernel = kernel[0]
    for weight in pytorch_state:
        if weight.startswith('weight'):
            pytorch_state[weight] = torch.tensor(kernel.transpose(2, 1, 0))
        elif weight.startswith('bias'):
            pytorch_state[weight] = torch.tensor(bias)
        else:
            raise ValueError("Unknown parameter {}".format(weight))
    pytorch_layer.load_state_dict(pytorch_state)
    other_info = {
        'activation': keras_config['activation'],
        'padding': keras_config['padding']
    }
    return pytorch_state, pytorch_config, other_info


def convert_keras2pytorch(keras_model):
    pytorch_config = []
    for layer in keras_model.layers:
        name = layer.name.lower()
        if 'conv1d' in name:
            pytorch_config.append(
                ('Conv1d', convert_Conv1d_layer(layer))
            )
        elif 'lstm' in name:
            pytorch_config.append(
                ('LSTM', convert_LSTM_layer(layer))
            )
    return pytorch_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--keras_model', type=str,
                        help='Path to Keras model (.h5)')
    parser.add_argument('--pytorch_model', type=str,
                        help='Path to output Pytorch model (.pt)')
    args, unknown = parser.parse_known_args(sys.argv[1:])
    if len(unknown) != 0:
        raise ValueError('Unknown argument {}\n'.format(unknown[0]))

    keras_model = load_keras_model(args.keras_model)
    pytorch_config = convert_keras2pytorch(keras_model)
    torch.save(pytorch_config, args.pytorch_model)
