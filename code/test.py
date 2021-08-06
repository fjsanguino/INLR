import numpy as np
import torch
from IQA_models.IQA_models import IQA_models_class, IQA_models_array


def avg(l):
    sum = 0
    for i in l:
        sum += i
    return sum / len(l)


def evaluate(model, data_loader):
    ''' set model to evaluate mode '''
    model.eval()
    with torch.no_grad():  # do not need to calculate information for gradient during eval
        metrics = {}
        for IQA_model in IQA_models_array:
            metrics[IQA_model] = list()

        for idx, (imgs) in enumerate(data_loader):
            imgs = imgs.cuda()
            pred = model(imgs)

            for IQA_model in IQA_models_array:
                IQA_metric_model = IQA_models_class(IQA_model)
                IQA_metric_model = IQA_metric_model.double().cuda()
                if IQA_model == 'MAE':
                    metrics[IQA_model].append(IQA_metric_model(pred, imgs).cpu().numpy().ravel())
                else:
                    metrics[IQA_model].append(IQA_metric_model(pred, imgs, as_loss=False).cpu().numpy())

    metrics_avg = {}
    for IQA_model in IQA_models_array:
        metrics[IQA_model] = np.concatenate(metrics[IQA_model])
        metrics_avg[IQA_model] = avg(metrics[IQA_model])

    return metrics_avg
