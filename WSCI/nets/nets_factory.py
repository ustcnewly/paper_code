import functools
import tensorflow as tf
from inception_v3 import inception_v3
from inception_v3 import inception_v3_arg_scope

slim = tf.contrib.slim

networks_map = { 'inception_v3': inception_v3}
arg_scopes_map = { 'inception_v3': inception_v3_arg_scope}


def get_network_fn(name, num_classes, dropout_train=True, reuse=None, weight_decay=0.0, batchnorm_train=True):
    """Returns a network_fn such as `logits, end_points = network_fn(images)`.

    Args:
        name: The name of the network.
        num_classes: The number of classes to use for classification.
        weight_decay: The l2 coefficient for the model weights.
        is_training: `True` if the model is being used for training and `False`
            otherwise.

    Returns:
        network_fn: A function that applies the model to a batch of images. It has
            the following signature:
                logits, end_points = network_fn(images)
    Raises:
        ValueError: If network `name` is not recognized.
    """
    
    if name not in networks_map:
        raise ValueError('Name of network unknown %s' % name)
    func = networks_map[name]
    @functools.wraps(func)
    def network_fn(images):
        arg_scope = arg_scopes_map[name](weight_decay=weight_decay)
        with slim.arg_scope(arg_scope):
            if name.startswith('resnet') or name.startswith('inception'): # networks using batch_norm
                return func(images, num_classes, reuse=reuse, is_training=dropout_train, moving_flag=batchnorm_train)
            else:
                return func(images, num_classes, reuse=reuse,  is_training=dropout_train)
    if hasattr(func, 'default_image_size'):
        
        network_fn.default_image_size = func.default_image_size

    return network_fn
