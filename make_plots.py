"""
Example usage: python make_plots.py --net vgg16 \
               --logs_file 'faster_rcnn_2_10_1251.txt'
"""

import argparse
import matplotlib.pyplot as plt
import matplotlib.style as style
import numpy as np
import os
style.use('ggplot')


def parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser(
        description='Plot Faster R-CNN losses from logs')

    parser.add_argument('--net', dest='net',
                        help='backbone network',
                        choices=['vgg16', 'res101'],
                        default='vgg16', type=str)
    parser.add_argument('--dataset', dest='dataset',
                        help='training dataset',
                        default='pascal_voc', type=str)
    parser.add_argument('--logs_file', dest='logs_file',
                        help='name of file to read logs from',
                        default='faster_rcnn_2_10_1251.txt', type=str)
    args = parser.parse_args()
    return args


def read_and_plot(net, dataset, logs_file, do_plot=False):
    """Read from logs and plot various Faster R-CNN losses over epochs"""
    logs_dir = os.path.join("logs", net, dataset)
    file_object = open(os.path.join(logs_dir, logs_file), "r")
    lines = file_object.readlines()

    # Read from logs
    results = []
    for l in lines[1:]:
        line = l.split(",")
        line[-1] = line[-1].replace("\n", "")
        line = [float(l_) for l_ in line]
        results.append(line)

    results = np.array(results)
    epochs = results[:, 0]
    losses = results[:, 1:]
    labels = ("Total loss", "Loss on RPN box regressor",
              "Loss on RPN classifier", "Loss on RCNN classifier",
              "Loss on RCNN box regressor")

    # Plot losses and save image to same location as logs file
    line_objects = plt.plot(epochs, losses, )
    plt.legend(line_objects, labels)
    plt.ylim(0, 5)
    plt.xlabel("Epoch number")
    plt.ylabel("Loss")
    plt.title(f"Losses with {net} backend")
    plt.tight_layout()
    session_id = logs_file[12:14]
    plt.savefig(os.path.join("plots",
                             f'{session_id}_{net}_loss.png'))
    if do_plot:
        plt.show()


if __name__ == '__main__':
    args = parse_args()
    read_and_plot(args.net, args.dataset, args.logs_file)
