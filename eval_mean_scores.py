import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import torch.backends.cudnn as cudnn

from torch import optim
from optparse import OptionParser
from torch.autograd import Variable

from myloss import dice_coeff
from utils import *
from models.unet import UNet

def eval_net(net, dataset, gpu=False):
    tot = 0
    mean_iou = 0
    mean_precision = 0
    mean_recall = 0
    for i, b in enumerate(dataset):
        X = b[0]
        y = b[1]

        X = torch.FloatTensor(X).unsqueeze(0)
        y = torch.ByteTensor(y).unsqueeze(0)

        if gpu:
            X = Variable(X, volatile=True).cuda()
            y = Variable(y, volatile=True).cuda()
        else:
            X = Variable(X, volatile=True)
            y = Variable(y, volatile=True)

        y_pred = net(X)

        y_pred = (F.sigmoid(y_pred) > 0.6).float()

        dice = dice_coeff(y_pred, y.float()).data[0]
        tot += dice

        # print('y_pred.data shape: ', y_pred.data.squeeze(0).squeeze(0).cpu().numpy().shape)
        # print('y.data: ', y.data.squeeze(0).cpu().numpy().shape)
        y = y.data.squeeze(0).cpu().numpy()
        y = (y > 0.6).astype(np.float32)
        y_pred = y_pred.data.squeeze(0).squeeze(0).cpu().numpy()
        y_pred = (y_pred > 0.5).astype(np.float32)
        # print('y.shape: ', y.shape)
        # print('y_pred.shape: ', y_pred.shape)

        # print('y binary : ', np.array_equal(y, y.astype(bool)))
        # print('y_pred binary : ', np.array_equal(y_pred, y_pred.astype(bool)))
        iou_val = get_iou(y_pred, y)
        mean_iou += iou_val
        precision_val = get_precision(y_pred, y)
        mean_precision += precision_val
        recall_val = get_recall(y_pred, y)
        mean_recall += recall_val

        if 0:
            X = X.data.squeeze(0).cpu().numpy()
            X = np.transpose(X, axes=[1, 2, 0])
            # y = y.data.squeeze(0).cpu().numpy()
            # y_pred = y_pred.data.squeeze(0).squeeze(0).cpu().numpy()

            fig = plt.figure()
            ax1 = fig.add_subplot(1, 4, 1)
            ax1.imshow(X)
            ax2 = fig.add_subplot(1, 4, 2)
            ax2.imshow(y)
            ax3 = fig.add_subplot(1, 4, 3)
            ax3.imshow(y_pred)

            Q = dense_crf(((X * 255).round()).astype(np.uint8), y_pred)
            ax4 = fig.add_subplot(1, 4, 4)
            print(Q)
            ax4.imshow(Q > 0.5)
            plt.show()
    print("i=",i)
    return tot / i, mean_iou / (i+1.0), mean_precision / (i+1.0), mean_recall / (i+1.0)




def Dice_Coeff(net, batch_size=2, lr=0.02, val_percent=1,
              cp=True, gpu=False):
    dir_img = '/media/markytools/503f6b96-90ca-4bfb-a99e-35f774205c77/EEE298/eee298deeplearning-finalproject-withdataset/LABELS_ONE_X/'
    dir_mask = '/media/markytools/503f6b96-90ca-4bfb-a99e-35f774205c77/EEE298/eee298deeplearning-finalproject-withdataset/LABELS_ONE_Y/'
    # dir_img = '/home/wdh/DataSets/hand-segmentation/GTEA_gaze_part/Resize/Images/'
    # dir_mask = '/home/wdh/DataSets/hand-segmentation/GTEA_gaze_part/Resize/Masks_1/'

    ids = get_ids(dir_img)
    ids = split_ids(ids)

    iddataset = split_train_val(ids, val_percent)

    print('''
    Starting evaluating:
        Batch size: {}
        Learning rate: {}
        Training size: {}
        Validation size: {}
        CUDA: {}
    '''.format(batch_size, lr, len(iddataset['train']),
               len(iddataset['val']),  str(gpu)))

    N_train = len(iddataset['train'])


    criterion = nn.BCELoss()


    # reset the generators
    val = get_imgs_and_masks(iddataset['val'], dir_img, dir_mask)

    if 1:
        evaluate_dice, mean_iou, mean_precision, mean_recall = eval_net(net, val, gpu)
        print('Dice Coeff for all: {}'.format(evaluate_dice))
        print('Mean IoU for all: {}'.format(mean_iou))
        print('Mean Precision for all: {}'.format(mean_precision))
        print('Mean Recall for all: {}'.format(mean_recall))


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-b', '--batch-size', dest='batchsize', default=2,
                      type='int', help='batch size')
    parser.add_option('-l', '--learning-rate', dest='lr', default=0.02,
                      type='float', help='learning rate')
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu',
                      default=False, help='use cuda')
    parser.add_option('-c', '--load', dest='load',
                      default=False, help='load file model')

    (options, args) = parser.parse_args()

    net = UNet(3, 1)

    if options.load:
        net.load_state_dict(torch.load(options.load))
        print('Model loaded from {}'.format(options.load))

    if options.gpu:
        net.cuda()
        cudnn.benchmark = True

    try:
        Dice_Coeff(net,options.batchsize, options.lr,
                  gpu=options.gpu)
    except KeyboardInterrupt:
        print('interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
