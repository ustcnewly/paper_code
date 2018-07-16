**Environment:**

* tensorflow 1.2.0, python 2.7

**Preparation:**

* Use 'datasets/generate_tfrecord_datasets.py' to generate tfrecord file for training and testing data. An example of the required image list is 'datasets/sample_image_list', in which each line contains image_path+' '+label_id. If there are in total C categories, the label_id is in [0, C-1].

* Prepare the semantic information for all the categories and generate 'attr2class.txt' with each column being the semantic information for each category.

* Place 'train.tfrecord', 'test.tfrecord', and 'attr2class.txt' in the folder 'datasets'.

* Modify the number of training/testing samples and the number of categories accordingly in 'datasets/dataset_factory.py'.

* Download the pretrained CNN model 'inception_v3.ckpt' from https://drive.google.com/open?id=15D8oEfXdMZ15iJTNjPnGwQK0yikPPkOr or https://pan.baidu.com/s/1ilOMiWZmgWa0PFx-0VoKiw, and place it in the folder 'pretrained_models'.

**Run:**

* Run 'python demo_run_WSCI.py' or with parameters 'python demo_run_WSCI.py --init_lr 0.001000 --beta 0.010000 --gpu_id 0'

