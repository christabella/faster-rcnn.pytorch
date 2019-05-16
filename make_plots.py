import matplotlib.pyplot as plt
import numpy as np


def read_and_plot(image_name_file_name,
                  file_path = "res101/pascal_voc/",
                  file_name="faster_rcnn_2_10_1251.txt"):
    file_object = open(file_path+file_name, "r")
    lines = file_object.readlines()

    results = []
    for l in lines[1:]:
        line = l.split(",")
        line[-1] = line[-1].replace("\n","")
        line = [float(l_) for l_ in line]
        results.append(line)
    results = np.array(results)

    plt.plot(results[:, 0], results[:, 1], label="Total Loss")
    plt.plot(results[:, 0], results[:, 2], label="Loss on RPN box regressor")
    plt.plot(results[:, 0], results[:, 3], label="Loss on RPN classifier")
    plt.plot(results[:, 0], results[:, 4], label="Loss on RCNN classfier")
    plt.plot(results[:, 0], results[:, 5], label="Loss on RCNN box regressor")
    plt.ylim(0,5)
    # epoch,loss,loss_rpn_cls,loss_rpn_box,loss_rcnn_cls,loss_rcnn_box
    plt.xlabel("Epoch number")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(image_name_file_name)
    #plt.show()


read_and_plot(image_name_file_name='res101_frcnn_loss.png',
              file_path="res101/pascal_voc/",
              file_name="faster_rcnn_2_10_1251.txt")
