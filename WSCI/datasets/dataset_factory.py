import os
import tensorflow as tf
import dataset_utils

slim = tf.contrib.slim

def get_sample_num(split_name):
    if split_name.startswith('train'):
        return 358500
    elif split_name.startswith('test'):
        return 14340
    else:
        raise ValueError('Name should start with train or test' % split_name)
    
_NUM_CLASSES = 717

_ITEMS_TO_DESCRIPTIONS = {
        'image': 'A color image of varying size.',
        'label': 'A single integer between 0 and 716'
}

def get_dataset(split_name, dataset_dir):
    """Given a dataset name and a split_name returns a Dataset.

    Args:
        split_name: A train/test split name.
        dataset_dir: The directory where the dataset files are stored.
        file_pattern: The file pattern to use for matching the dataset source files.

    Returns:
        A `Dataset` class.
    """
    
    
    file_pattern = os.path.join(dataset_dir, split_name+'.tfrecord')
    reader = tf.TFRecordReader

    keys_to_features = {
            'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
            'image/format': tf.FixedLenFeature((), tf.string, default_value='png'),
            'image/class/label': tf.FixedLenFeature(
                    [], tf.int64, default_value=tf.zeros([], dtype=tf.int64))
    }

    items_to_handlers = {
            'image': slim.tfexample_decoder.Image(),
            'label': slim.tfexample_decoder.Tensor('image/class/label')
    }

    decoder = slim.tfexample_decoder.TFExampleDecoder(
            keys_to_features, items_to_handlers)

    labels_to_names = None
    if dataset_utils.has_labels(dataset_dir):
        labels_to_names = dataset_utils.read_label_file(dataset_dir)

    return slim.dataset.Dataset(
            data_sources=file_pattern,
            reader=reader,
            decoder=decoder,
            num_samples=get_sample_num(split_name),
            items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
            num_classes=_NUM_CLASSES,
            labels_to_names=labels_to_names)
