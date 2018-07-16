import os
import math
import numpy as np
import tensorflow as tf
from datasets import dataset_factory
from nets import nets_factory
from WSCI_network import WSCI_network
from utils import load_batch, configure_optimizer, configure_learning_rate, get_variables_to_train, get_init_fn

tf.app.flags.DEFINE_integer( 'gpu_id', 0, 'The GPU card to use.')
tf.app.flags.DEFINE_float( 'init_lr', 1e-3, 'The initial learning rate.')
tf.app.flags.DEFINE_float( 'beta', 1e-3, 'The coefficient for VAE reconstruction.')

slim = tf.contrib.slim
FLAGS = tf.app.flags.FLAGS

      
def main(_):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu_id)
    FLAGS.num_preprocessing_threads = 10
    FLAGS.max_epoch_num = 100
    FLAGS.train_split_name='train'
    FLAGS.test_split_name='test'    
    FLAGS.model_name='inception_v3'
    
    FLAGS.dataset_dir = 'datasets'
    FLAGS.attr2class_file = os.path.join(FLAGS.dataset_dir, 'attr2class.txt')
    FLAGS.train_dir='output' 
    FLAGS.checkpoint_path=os.path.join('pretrained_models','%s.ckpt' %(FLAGS.model_name))
    log_file_path = os.path.join(FLAGS.train_dir, 'log')
    
    if not os.path.isdir(FLAGS.train_dir):
        os.makedirs(FLAGS.train_dir)
        
    FLAGS.checkpoint_exclude_scopes= 'InceptionV3/Logits,InceptionV3/AuxLogits,VAE'
    FLAGS.trainable_scopes='VAE' # if learning all parameters including CNN, set FLAGS.trainable_scopes=None
    FLAGS.checkpoint_exclude_keywords = None 
    FLAGS.batch_size=64

    with tf.Graph().as_default():
        # load dataset
        dataset = dataset_factory.get_dataset(FLAGS.train_split_name, FLAGS.dataset_dir)
        test_dataset = dataset_factory.get_dataset(FLAGS.test_split_name, FLAGS.dataset_dir)
        num_batches = int(math.ceil(dataset.num_samples / float(FLAGS.batch_size)))
        num_test_batches = int(math.ceil(test_dataset.num_samples / float(FLAGS.batch_size)))
        train_image_size = nets_factory.get_network_fn(FLAGS.model_name, dataset.num_classes).default_image_size        
        images, labels = load_batch(FLAGS, dataset, train_image_size, train_image_size, is_training=True)        
        test_images, test_labels = load_batch(FLAGS, test_dataset, train_image_size, train_image_size, is_training=False) 
                      
        # load class attributes
        attr2class = np.loadtxt(FLAGS.attr2class_file, np.float32)
        
        # build networks
        train_batch_loss, train_summary = WSCI_network('train', FLAGS.model_name, images, labels,  attr2class, False,  FLAGS.beta)
        test_correct_arr = WSCI_network('test', FLAGS.model_name, test_images, test_labels,  attr2class,  True)

        # optimizer
        global_step = slim.create_global_step()
        config_lr = configure_learning_rate(FLAGS, dataset.num_samples, global_step)
        optimizer = configure_optimizer( learning_rate=config_lr)        
        variables_to_train = get_variables_to_train(FLAGS)
        grads_and_vars = optimizer.compute_gradients(train_batch_loss,  variables_to_train)
        update_ops = [optimizer.apply_gradients(grads_and_vars, global_step=global_step)]
        # update moving_mean and moving_variance for batch normalization
        # update_ops += tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        update_op = tf.group(*update_ops)
        with tf.control_dependencies([update_op]):
            train_op = tf.identity(train_batch_loss)        
        
        # main code
        config = tf.ConfigProto() 
        config.gpu_options.allow_growth=True 

        with tf.Session(config=config) as sess:        
            # initialization
            sess.run(tf.global_variables_initializer())
            
            with slim.queues.QueueRunners(sess):
                # initialization
                iepoch = 0                
                init_fn=get_init_fn(FLAGS.checkpoint_path, FLAGS.checkpoint_exclude_scopes, FLAGS.checkpoint_exclude_keywords)
                init_fn(sess)
                
                while iepoch<FLAGS.max_epoch_num:    
                    #training
                    for ibatch in  range(num_batches):
                        print 'iepoch %d: train %d/%d' %(iepoch, ibatch, num_batches)           
                        sess.run([train_op, train_summary])      

                    # test
                    correct_num = 0
                    for ibatch  in range(num_test_batches):
                        print 'iepoch %d: test %d/%d' %(iepoch, ibatch, num_test_batches)
                        correct_arr = sess.run(test_correct_arr)           
                        correct_num += np.sum(correct_arr)
                    test_acc = float(correct_num)/(num_test_batches*FLAGS.batch_size)                                                          
                    
                    fid = open(log_file_path, 'a+')
                    fid.write('%d %f\n' %(iepoch, test_acc))
                    fid.close()                        
                                                
                
if __name__ == '__main__':
    tf.app.run()
