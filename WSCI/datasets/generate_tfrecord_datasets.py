import os
import dataset_utils
import tensorflow as tf

import pandas as pd

class ImageReader(object):
    """Helper class that provides TensorFlow image coding utilities."""

    def __init__(self):
        # Initializes function that decodes RGB JPEG data.
        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

    def read_image_dims(self, sess, image_data):
        image = self.decode_jpeg(sess, image_data)
        return image.shape[0], image.shape[1]

    def decode_jpeg(self, sess, image_data):
        image = sess.run(self._decode_jpeg,
                                         feed_dict={self._decode_jpeg_data: image_data})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image
    
def _convert_dataset(specs_name, filenames, class_ids):
    with tf.Graph().as_default():
        image_reader = ImageReader()
        config = tf.ConfigProto() 
        config.gpu_options.allow_growth=True 
        with tf.Session(config=config) as sess:
            output_filename = specs_name+'.tfrecord'
            if os.path.exists(output_filename):
                os.remove(output_filename)

            with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
                for i in range(len(filenames)):
                    # Read the filename:
                    print str(i)+' '+ filenames[i]
                    img_path = filenames[i]
                    image_data = tf.gfile.FastGFile(img_path, 'rb').read()
                    height, width = image_reader.read_image_dims(sess, image_data)
                    class_id = int( class_ids[i])

                    example = dataset_utils.image_to_tfexample(
                            image_data, b'jpg', height, width, class_id)
                    tfrecord_writer.write(example.SerializeToString())


if __name__ == "__main__":
           
    image_list_file = 'sample_image_list'
    specs_name  = 'train'
   
    csv_data = pd.read_csv(image_list_file,  delimiter=' ', header=None)    
    filenames = csv_data.loc[:,0].tolist()
    class_ids = csv_data.loc[:,1].tolist()    
    
    _convert_dataset(specs_name, filenames, class_ids)
    

    
