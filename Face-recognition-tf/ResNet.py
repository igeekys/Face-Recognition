import os
import time

import tensorflow as tf
import tf_slim as slim
from tf_slim.nets import resnet_v1

from ops import (batch_norm, bottle_resblock, classification_loss, conv,
                 fully_conneted, get_residual_layer, global_avg_pooling, relu,
                 resblock)
from resnet_v2_SE import SE_Inception_resnet_v2
from utils import LoadDataset, data_augmentation


class ResNet(object):
    def __init__(self, sess, args):
        
        self.sess = sess
        self.dataset_path = args.dataset
        self.split_dataset = args.split_dataset

        self.img_size = 224
        self.c_dim = 3
        self.Nclass = 4
        
        
        self.checkpoint_dir = args.checkpoint_dir
        self.log_dir = args.log_dir

        self.res_n = args.res_n

        self.epoch = args.epoch
        self.batch_size = args.batch_size

        self.init_lr = args.lr
        self.pretrained_model = args.pretrained_model
        self.pretrained = args.pretrained
        self.if_SE = args.if_SE
        if self.if_SE:
            self.model_name = 'ResNet-SE'
        else:
            self.model_name = 'ResNet'

    ##################################################################################
    # Model
    ##################################################################################

    def build_model(self):
        """ Graph Input """
        self.inputs = tf.placeholder(tf.float32, [None, self.img_size, self.img_size, self.c_dim], name='inputs')
        self.labels = tf.placeholder(tf.float32, [None, self.Nclass], name='labels')
        self.lr = tf.placeholder(tf.float32, name='learning_rate')
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        
        if self.if_SE:
            logits = SE_Inception_resnet_v2(self.inputs, self.Nclass, is_training=self.is_training).model
        else:
            with slim.arg_scope(resnet_v1.resnet_arg_scope()):
                net, endpoints = resnet_v1.resnet_v1_50(self.inputs, num_classes=None, is_training=self.is_training)
            with tf.variable_scope('Logits'):
                net = tf.squeeze(net, axis=[1, 2])
                net = slim.dropout(net, keep_prob=0.5, scope='scope')
                logits = slim.fully_connected(net, num_outputs=self.Nclass, activation_fn=None, scope='fc')
                    
        self.loss, self.accuracy,self.top_k_acc = classification_loss(logits, self.labels, top_k=2)
        
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.train_step = self.optimizer.minimize(self.loss)


    ##################################################################################
    # Train
    ##################################################################################

    def train(self):

        init = tf.global_variables_initializer()

        checkpoint_exclude_scopes = 'Logits'
        exclusions = None
        if checkpoint_exclude_scopes:
            exclusions = [scope.strip() for scope in checkpoint_exclude_scopes.split(',')]
        variables_to_restore = []
        for var in slim.get_model_variables():
            excluded = False
            for exclusion in exclusions:
                if var.op.name.startswith(exclusion):
                    excluded = True
            if not excluded:
                variables_to_restore.append(var)
        self.saver_restore = tf.train.Saver(var_list=variables_to_restore)
        self.saver = tf.train.Saver(tf.global_variables())
        # summary writer
        # self.writer = tf.summary.FileWriter(os.path.join(self.log_dir, self.model_dir), self.sess.graph)

        # config = tf.ConfigProto(allow_soft_placement = True) 
        # config.gpu_options.per_process_gpu_memory_fraction = 0.95
        # with tf.Session(config=config) as sess:
        self.sess.run(init)
        
        # Load the pretrained checkpoint file xxx.ckpt
        # ckpt = tf.train.get_checkpoint_state('/Users/michael/Desktop/4.4-1400/tf-code/ResNet/checkpoint/') 
        # if ckpt and ckpt.model_checkpoint_path:
        #     self.saver_restore.restore(self.sess, ckpt.model_checkpoint_path)
        if tf.train.latest_checkpoint('./checkpoint/'):
            self.saver_restore.restore(self.sess, tf.train.latest_checkpoint('./checkpoint/'))
            print("RESTORE MODEL SUCCESSFULLY")
        self.train_x, self.train_y, self.valid_x, self.valid_y, self.test_x, self.test_y = LoadDataset(self)
        self.iteration = len(self.train_x) // self.batch_size
        start_time = time.time()
        epoch_lr = self.init_lr
        start_epoch = 0
        start_batch_id = 0
        counter = 1
        for epoch in range(self.epoch):
            if epoch == int(self.epoch * 0.5) or epoch == int(self.epoch * 0.75) :
                epoch_lr = epoch_lr * 0.1

            # get batch data
            for idx in range(start_batch_id, self.iteration):
                batch_x = self.train_x[idx*self.batch_size : (idx+1)*self.batch_size]
                batch_y = self.train_y[idx*self.batch_size : (idx+1)*self.batch_size]

                batch_x = data_augmentation(batch_x, self.img_size)

                train_feed_dict = {
                    self.inputs : batch_x,
                    self.labels : batch_y,
                    self.lr : epoch_lr,
                    self.is_training:True
                }
                valid_feed_dict = {
                    self.inputs : self.valid_x,
                    self.labels : self.valid_y,
                    self.is_training: False
                }
                # update network
                _, train_loss, train_accuracy, train_topk_acc = self.sess.run(
                    [self.train_step, self.loss, self.accuracy, self.top_k_acc], feed_dict=train_feed_dict)
                # self.writer.add_summary(summary_str, counter)

                # validate
                valid_loss, valid_accuracy, valid_topk_acc = self.sess.run(
                    [self.loss, self.accuracy, self.top_k_acc], feed_dict=valid_feed_dict)
                # self.writer.add_summary(summary_str, counter)

                # display training status
                counter += 1
                print("Epoch: [{:2d}] [{:5d}/{:5d}] time: {:4.4f}, learning_rate : {:.4f} \ntrain_loss: {:.2f},  train_accuracy: {:.2f},  train_topk_acc: {:.2f},   valid_loss: {:.2f},  valid_accuracy: {:.2f},  valid_topk_acc: {:.2f}".format(
                      epoch, idx, self.iteration, time.time() - start_time, epoch_lr, train_loss, train_accuracy, train_topk_acc, valid_loss, valid_accuracy, valid_topk_acc))

            # After an epoch, start_batch_id is set to zero
            # non-zero value is only for the first epoch after loading pre-trained model
            start_batch_id = 0

            # save model
            self.save(self.checkpoint_dir, counter)

        # save model for final step
        self.save(self.checkpoint_dir, counter)


    @property
    def model_dir(self):
        return "{}{}_{}_{}".format(self.model_name, self.res_n, self.batch_size, self.init_lr)

    def save(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, self.model_name+'.model'), global_step=step)


    def test(self):
        tf.global_variables_initializer().run()

        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, tf.train.latest_checkpoint(os.path.join(self.checkpoint_dir, self.model_dir)))

        test_feed_dict = {
            self.inputs: self.test_x,
            self.labels: self.test_y,
            self.is_training: False
        }

        test_accuracy, test_top_k = self.sess.run([self.accuracy,self.top_k_acc], feed_dict=test_feed_dict)
        print("test_accuracy: {:.2f},  test_top_k_acc: {:.2f}".format(test_accuracy, test_top_k))
