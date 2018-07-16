import tensorflow as tf
import math
from nets import nets_factory

slim = tf.contrib.slim

def WSCI_network(stage, model_name,  input_images, onehot_labels,  attr2class, reuse=False,   beta=1.0):
        
    class_num = attr2class.shape[1]
    if stage=='train':
        network_fn = nets_factory.get_network_fn(model_name, class_num, dropout_train=False, batchnorm_train=False)
    else:
        network_fn = nets_factory.get_network_fn(model_name, class_num, dropout_train=False, batchnorm_train=False,  reuse=reuse)
        
    _, end_points = network_fn(input_images)
    features = end_points['PreLogits'] 
    
    # variational auto encoder                        
    return VAE_model(stage, features, onehot_labels, attr2class,  reuse, beta)
        
        
def VAE_model(stage, x, onehot_labels, attr2class, reuse,  beta):

        attr_dim = attr2class.shape[0]
        attr2class = tf.convert_to_tensor(attr2class)
                
        with tf.variable_scope('VAE', reuse=reuse):
            weights = dict()
            summary = dict()
            initw = tf.truncated_normal_initializer(0.0, 0.001)
            initb = tf.constant_initializer(0.0)
            pi = tf.constant(math.pi)
            
            x = tf.squeeze(x, [1, 2])
            batch_size = int(x.shape[0])
            feat_dim = int(x.shape[1])
   
            encoder_hidden_dim = (feat_dim+attr_dim)/2
            decoder_hidden_dim = (feat_dim+attr_dim)/2
           
            with tf.variable_scope('encoder'):
                # encoder weights
                weights['encoder_hidden_w'] = tf.get_variable("encoder_hidden_w", shape=[feat_dim, encoder_hidden_dim], initializer=initw)
                weights['encoder_hidden_b'] = tf.get_variable("encoder_hidden_b", shape=[encoder_hidden_dim], initializer=initb)
                            
                weights['encoder_mean_w'] = tf.get_variable("encoder_mean_w", shape=[encoder_hidden_dim, attr_dim], initializer=initw)
                weights['encoder_mean_b'] = tf.get_variable("encoder_mean_b", shape=[attr_dim], initializer=initb)
                weights['encoder_var_w'] = tf.get_variable("encoder_var_w", shape=[encoder_hidden_dim, attr_dim], initializer=initw)
                weights['encoder_var_b'] = tf.get_variable("encoder_var_b", shape=[attr_dim], initializer=initb)
            # decoder weights
            with tf.variable_scope('decoder'):
                weights['decoder_hidden_w'] = tf.get_variable("decoder_hidden_w", shape=[attr_dim, decoder_hidden_dim], initializer=initw)
                weights['decoder_hidden_b'] = tf.get_variable("decoder_hidden_b", shape=[decoder_hidden_dim], initializer=initb)            
                weights['decoder_mean_w'] = tf.get_variable("decoder_mean_w", shape=[decoder_hidden_dim, feat_dim], initializer=initw)
                weights['decoder_mean_b'] = tf.get_variable("decoder_mean_b", shape=[feat_dim], initializer=initb)

            h_1 = tf.nn.relu(tf.matmul(x, weights['encoder_hidden_w']) + weights['encoder_hidden_b'])
            z_mu = tf.add(tf.matmul(h_1, weights['encoder_mean_w']), weights['encoder_mean_b'])
         
            #  the network model the parameter log(\sigma^2) $\in [\infty, \infty]$
            z_ls2 = tf.add(tf.matmul(h_1, weights['encoder_var_w']), weights['encoder_var_b'])
            eps = tf.random_normal((batch_size, attr_dim), 0, 1, dtype=tf.float32) 
            # sample z using reparameterization trick
            z = z_mu + tf.sqrt(tf.exp(z_ls2))* eps  

            # decision values based on latent embeddings
            class_logits = tf.matmul(z, attr2class)
            pred_label = tf.arg_max(class_logits, 1)
            gt_label = tf.arg_max(onehot_labels, 1)
            correct_arr = tf.cast(tf.equal(pred_label, gt_label), tf.float32)
        
            h_1_g  = tf.nn.relu(tf.matmul(z, weights['decoder_hidden_w']) + weights['decoder_hidden_b'])                
            x_mu = tf.add(tf.matmul(h_1_g,  weights['decoder_mean_w']), weights['decoder_mean_b'])
            # fix x_variance as 1 for stability
            
            nonoutlier_log_prob =  -tf.reduce_sum(0.5 * tf.log(2*pi) + tf.square(x-x_mu)/2.0, 1)
            nonoutlier_prob = tf.exp(nonoutlier_log_prob - tf.reduce_max(nonoutlier_log_prob))
            nonoutlier_prob = tf.pow(nonoutlier_prob+1e-30, 0.01)
            
            # weighted classification loss
            cls_loss = tf.losses.softmax_cross_entropy(onehot_labels, class_logits, reduction='none')
            weighted_cls_loss = tf.reduce_sum(nonoutlier_prob*cls_loss)
            # reconstruction loss
            reconstruction_loss = tf.reduce_sum(-nonoutlier_log_prob)
            total_loss = weighted_cls_loss + beta*reconstruction_loss
            
            summary['weighted_cls_loss'] = weighted_cls_loss;
            summary['reconstruction_loss'] = reconstruction_loss
            
            if stage=='train':
                return total_loss, summary
            else:
                return correct_arr
                                              
            
        