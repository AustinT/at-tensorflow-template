import tensorflow as tf


def get_layer(layer_num, last_layer, layer_type, name=None, **kwargs):
    if name is None:
        name = "Hidden{}-{}".format(layer_num, layer_type)
    if layer_type == "conv2d":
        return tf.layers.conv2d(last_layer, name=name, **kwargs)
    elif layer_type == "conv2dT":
        return tf.layers.conv2d_transpose(last_layer, name=name, **kwargs)
    elif layer_type == "maxpool":
        return tf.layers.max_pooling2d(last_layer, name=name, **kwargs)
    elif layer_type == "avgpool":
        return tf.layers.average_pooling2d(last_layer, name=name, **kwargs)
    elif layer_type == "flatten":
        return tf.layers.flatten(last_layer, name=name, **kwargs)
    elif layer_type == "dense":
        return tf.layers.dense(last_layer, name=name, **kwargs)
    elif layer_type == "reshape":
        shape = kwargs.pop("shape")
        return tf.reshape(last_layer, shape, name=name, **kwargs)
    else:
        raise NotImplementedError
