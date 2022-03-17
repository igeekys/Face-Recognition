import argparse
import os
import tensorflow as tf

from ResNet import ResNet
from utils import check_folder, show_all_variables

"""parsing and configuration"""
def parse_args():
    desc = "Tensorflow implementation of ResNet"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--phase', type=str, default='train', help='train or test ?')
    parser.add_argument('--dataset', type=str, default='../dataset', help='dataset path')
    parser.add_argument('--split_dataset', action='store_true', 
                        help='split dataset into train/valid/test parts while the firt running')
    parser.add_argument('--epoch', type=int, default=10, help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=64, help='The size of batch per gpu')
    parser.add_argument('--res_n', type=int, default=50, help='18, 34, 50, 101, 152')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint',
                        help='Directory name to save the checkpoints')
    parser.add_argument('--pretrained', action='store_true', 
                        help='if restore from the pretrained model of Imagenet')
    parser.add_argument('--pretrained_model', type=str, default='checkpoint/resnet_v1_50.ckpt',
                        help='path of the pretrained checkpoint')
    parser.add_argument('--if_SE', action='store_true', 
                        help='if use SE to the model')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory name to save training logs')

    return check_args(parser.parse_args())

"""checking arguments"""
def check_args(args):
    # --checkpoint_dir
    check_folder(args.checkpoint_dir)

    # --result_dir
    check_folder(args.log_dir)

    # --epoch
    try:
        assert args.epoch >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')
    return args


"""main"""
def main():
    # parse arguments
    args = parse_args()
    if args is None:
      exit()

    # args.split_dataset = True
    args.pretrained = True
    args.if_SE = True
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # open session
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        cnn = ResNet(sess, args)

        # build graph
        cnn.build_model()

        # show network architecture
        show_all_variables()

        if args.phase == 'train' :
            # launch the graph in a session
            cnn.train()

            print(" [*] Training finished! \n")

            cnn.test()
            print(" [*] Test finished!")

        if args.phase == 'test' :
            cnn.test()
            print(" [*] Test finished!")

if __name__ == '__main__':
    main()
