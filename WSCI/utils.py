import tensorflow as tf
slim = tf.contrib.slim
from preprocessing import preprocessing_factory

def load_batch(FLAGS, dataset, height=299, width=299, is_training=True):
    preprocessing_name =  FLAGS.model_name
    image_preprocessing_fn = preprocessing_factory.get_preprocessing(
            preprocessing_name,
            is_training=False)

    provider = slim.dataset_data_provider.DatasetDataProvider(dataset, shuffle=is_training,
            common_queue_capacity=128 * FLAGS.batch_size, common_queue_min=64 * FLAGS.batch_size)
    
    [image, label] = provider.get(['image', 'label'])

    image = image_preprocessing_fn(image, height, width)
    images, labels = tf.train.batch([image, label], batch_size=FLAGS.batch_size, 
                                    num_threads=FLAGS.num_preprocessing_threads, capacity=(FLAGS.num_preprocessing_threads+2) * FLAGS.batch_size)
    labels = slim.one_hot_encoding(labels, dataset.num_classes)     
    batch_queue = slim.prefetch_queue.prefetch_queue([images, labels], capacity=2)
    images, labels = batch_queue.dequeue()
    return images, labels

def get_init_fn(checkpoint_path, checkpoint_exclude_scopes=None, checkpoint_exclude_keywords=None):
    """Returns a function run by the chief worker to warm-start the training.

    Note that the init_fn is only run when initializing the model during the very
    first global step.

    Returns:
        An init function run by the supervisor.
    """

    exclusions = []
    if checkpoint_exclude_scopes:
        exclusions = [scope.strip()  for scope in checkpoint_exclude_scopes.split(',')]

    keywords = []
    if checkpoint_exclude_keywords:
        keywords = [keyword.strip()   for keyword in checkpoint_exclude_keywords.split(',')]
        
    variables_to_restore = []
        
    model_variables = tf.get_collection(tf.GraphKeys.MODEL_VARIABLES)
    model_variables += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'VAE')        
            
    for var in model_variables:
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
                break
        for keyword in keywords:
            if keyword in var.op.name:
                excluded = True
                break
        if not excluded:
            variables_to_restore.append(var)

    if tf.gfile.IsDirectory(checkpoint_path):
        checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)
    else:
        checkpoint_path = checkpoint_path

    tf.logging.info('Fine-tuning from %s' % checkpoint_path)

    return slim.assign_from_checkpoint_fn(checkpoint_path, variables_to_restore)


def get_variables_to_train(FLAGS):
    """Returns a list of variables to train.

    Returns:
        A list of variables to train by the optimizer.
    """
    if FLAGS.trainable_scopes is None:
        return tf.trainable_variables()
    else:
        scopes = [scope.strip() for scope in FLAGS.trainable_scopes.split(',')]

    variables_to_train = []
    for scope in scopes:
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        variables_to_train.extend(variables)
    return variables_to_train


def configure_learning_rate(FLAGS,num_samples_per_epoch, global_step):

    decay_steps = int(num_samples_per_epoch / FLAGS.batch_size *2.0)
    return tf.train.exponential_decay(FLAGS.init_lr, global_step, decay_steps, 0.94, staircase=True, name='exponential_decay_learning_rate')
        
def configure_optimizer(learning_rate):
    optimizer = tf.train.AdamOptimizer(
            learning_rate,
            beta1=0.9,
            beta2=0.999,
            epsilon=1.0)
    return optimizer