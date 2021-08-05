import os
import torch

import parser
import models_CNN
import data

import numpy as np
import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter
from test import evaluate
from INRF_IQ.INRF_IQ import INRF_IQ


def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)


if __name__ == '__main__':

    args = parser.arg_parse()

    '''create directory to save trained model and other info'''
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    ''' setup GPU '''
    torch.cuda.set_device(args.gpu)

    ''' setup random seed '''
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)

    ''' load dataset and prepare data loader '''
    print('===> prepare dataloader ...')
    '''
    train_loader = data.MultiEpochsDataLoader(data.DATA(args, mode='train'),
                                               batch_size=args.train_batch,
                                               num_workers=args.workers,
                                               shuffle=True)
    val_loader = data.MultiEpochsDataLoader(data.DATA(args, mode='test'),
                                               batch_size=args.train_batch,
                                               num_workers=args.workers,
                                               shuffle=True)
    '''
    train_loader = torch.utils.data.DataLoader(data.DATA(args, mode='train'),
                                               batch_size=args.train_batch,
                                               num_workers=args.workers,
                                               shuffle=True)
    val_loader = torch.utils.data.DataLoader(data.DATA(args, mode='test'),
                                               batch_size=args.train_batch,
                                               num_workers=args.workers,
                                               shuffle=True)
    ''' load model '''
    print('===> prepare model ...')
    model = models_CNN.CompressionNetwork()
    model.cuda()  # load model to gpu

    ''' define loss '''
    if args.loss == 'MAE':
        criterion = nn.L1Loss() #MAE
    if args.loss == 'INRF-IQ':
        criterion = INRF_IQ()
    else:
        criterion = nn.L1Loss()

    ''' setup optimizer '''
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    ''' setup tensorboard '''
    writer = SummaryWriter(os.path.join(args.save_dir, args.loss))

    ''' train model '''
    print('===> start training ...')
    iters = 0
    for epoch in range(1, args.epoch + 1):

        model.train()

        for idx, (imgs) in enumerate(train_loader):

            train_info = 'Epoch: [{0}][{1}/{2}]'.format(epoch, idx + 1, len(train_loader))
            iters += 1

            ''' move data to gpu '''
            imgs = imgs.cuda()

            ''' forward path '''
            output = model(imgs)

            ''' compute loss, backpropagation, update parameters '''
            loss = criterion(output, imgs)  # compute loss

            optimizer.zero_grad()  # set grad of all parameters to zero
            loss.backward()  # compute gradient for each parameters
            optimizer.step()  # update parameters

            ''' write out information to tensorboard '''
            writer.add_scalar('loss', loss.data.cpu().numpy(), iters)
            train_info += ' loss: {:.4f}'.format(loss.data.cpu().numpy())

            print(train_info)

        if epoch % args.val_epoch == 0:
            ''' evaluate the model '''
            metrics = evaluate(model, val_loader)
            writer.add_scalars('metrics', metrics, iters)

            metrics = sorted(metrics.items(), key=lambda x: x[1])
            print('Epoch: [{}]'.format(epoch))
            for i, (metric, value) in enumerate(metrics):
                print('{}. {}: {}'.format(i + 1, metric, value))
            print('-----------------------------------------')

        ''' save model '''
        save_model(model, os.path.join(args.save_dir, 'model_{}.pth.tar'.format(epoch)))
