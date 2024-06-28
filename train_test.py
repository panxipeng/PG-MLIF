import random

from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.data import RandomSampler

from data_loaders import PathomicDatasetLoader, PathomicFastDatasetLoader
from networks import define_net, define_reg, define_optimizer, define_scheduler
from utils import unfreeze_unimodal, CoxLoss, CoxLoss2, CIndex_lifeline, cox_log_rank, accuracy_cox, mixed_collate, \
    count_parameters

# from GPUtil import showUtilization as gpu_usage
import pdb
import pickle
import os
import tensorflow as tf
# from tensorflow.keras.models import networks


def train(opt, data, device, k):
    cudnn.deterministic = True
    torch.cuda.manual_seed_all(2019)
    torch.manual_seed(2019)
    random.seed(2019)
    model = define_net(opt, k)
    optimizer = define_optimizer(opt, model)
    scheduler = define_scheduler(opt, optimizer)
    print(model)
    print("Number of Trainable Parameters: %d" % count_parameters(model))
    print("Activation Type:", opt.act_type)
    print("Optimizer Type:", opt.optimizer_type)
    print("Regularization Type:", opt.reg_type)

    use_patch, roi_dir = ('_patch_', 'all_st_patches_512') if opt.use_vgg_features else ('_', 'all_st')

    custom_data_loader = PathomicFastDatasetLoader(opt, data, split='train',
                                                        mode=opt.mode) if opt.use_vgg_features else PathomicDatasetLoader(
        opt, data, split='train', mode=opt.mode)
    train_loader = torch.utils.data.DataLoader(dataset=custom_data_loader, batch_size=opt.batch_size, shuffle=True,
                                               collate_fn=mixed_collate, drop_last=True)
    metric_logger = {'train': {'loss': [], 'pvalue': [], 'cindex': [], 'surv_acc': [], 'grad_acc': []},
                     'test': {'loss': [], 'pvalue': [], 'cindex': [], 'surv_acc': [], 'grad_acc': []}}
    yy_col = []
    y_col = []

    for epoch in tqdm(range(opt.epoch_count, opt.niter + opt.niter_decay + 1)):

        if opt.finetune == 1:
            unfreeze_unimodal(opt, model, epoch)

        model.train()
        risk_pred_all, censor_all, survtime_all = np.array([]), np.array([]), np.array([])  # Used for calculating the C-Index
        loss_epoch, grad_acc_epoch = 0, 0

        for batch_idx, (x_path, r_mul, x_omic, censor, survtime, grade) in enumerate(train_loader):

            censor = censor.to(device) if "surv" in opt.task else censor
            grade = grade.to(device) if "grad" in opt.task else grade
            _, pred = model(x_path=x_path.to(device), x_grph=r_mul.to(device), x_omic=x_omic.to(device))
            # print('_.shape', _.shape)  32*32
            # print('pred.shape', pred.shape)  32*1
            loss_cox = CoxLoss(survtime, censor, pred, device) if opt.task == "surv" else 0
            loss_reg = define_reg(opt, model)
            loss_nll = F.nll_loss(pred, grade) if opt.task == "grad" else 0
            loss = opt.lambda_cox * loss_cox + opt.lambda_nll * loss_nll + opt.lambda_reg * loss_reg
            loss_epoch += loss.data.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if opt.task == "surv":
                risk_pred_all = np.concatenate(
                    (risk_pred_all, pred.detach().cpu().numpy().reshape(-1)))  # Logging Information
                censor_all = np.concatenate(
                    (censor_all, censor.detach().cpu().numpy().reshape(-1)))  # Logging Information
                survtime_all = np.concatenate(
                    (survtime_all, survtime.detach().cpu().numpy().reshape(-1)))  # Logging Information

            if opt.verbose > 0 and opt.print_every > 0 and (
                    batch_idx % opt.print_every == 0 or batch_idx + 1 == len(train_loader)):
                print("Epoch {:02d}/{:02d} Batch {:04d}/{:d}, Loss {:9.4f}".format(
                    epoch + 1, opt.niter + opt.niter_decay, batch_idx + 1, len(train_loader), loss.item()))
        y_col.append(loss_epoch / len(train_loader))
        yy_col.append(y_col)
        scheduler.step()

        if opt.measure or epoch == (opt.niter + opt.niter_decay - 1):
            loss_epoch /= len(train_loader)

            cindex_epoch = CIndex_lifeline(risk_pred_all, censor_all, survtime_all) if opt.task == 'surv' else None
            pvalue_epoch = cox_log_rank(risk_pred_all, censor_all, survtime_all) if opt.task == 'surv' else None
            surv_acc_epoch = accuracy_cox(risk_pred_all, censor_all) if opt.task == 'surv' else None
            grad_acc_epoch = grad_acc_epoch / len(train_loader.dataset) if opt.task == 'grad' else None
            loss_test, cindex_test, pvalue_test, surv_acc_test, grad_acc_test, pred_test = test(opt, model, data,
                                                                                                'test', device)

            metric_logger['train']['loss'].append(loss_epoch)
            metric_logger['train']['cindex'].append(cindex_epoch)
            metric_logger['train']['pvalue'].append(pvalue_epoch)
            metric_logger['train']['surv_acc'].append(surv_acc_epoch)

            metric_logger['test']['loss'].append(loss_test)
            metric_logger['test']['cindex'].append(cindex_test)
            metric_logger['test']['pvalue'].append(pvalue_test)
            metric_logger['test']['surv_acc'].append(surv_acc_test)

            pickle.dump(pred_test, open(os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name,
                                                     '%s_%d%s%d_pred_test.pkl' % (opt.model_name, k, use_patch, epoch)), 'wb'))

            if opt.verbose > 0:
                if opt.task == 'surv':
                    print('[{:s}]\t\tLoss: {:.4f}, {:s}: {:.4f}'.format('Train', loss_epoch, 'C-Index', cindex_epoch))
                    print('[{:s}]\t\tLoss: {:.4f}, {:s}: {:.4f}\n'.format('Test', loss_test, 'C-Index', cindex_test))


    plt.figure(50)
    x_col = np.linspace(0, 30  , 30)  # 数组逆序
    plt.plot(x_col, y_col)
    plt.xlabel("epoch")
    plt.ylabel("loss_value")
    plt.title("loss", fontweight="semibold", fontsize='x-large')
    # plt.show()
    path = '/home/yons/An/paper/20220318-PathomicFusion/20230529PO-Fusion/2splits_15/20230608_pathomic_2splits_15'
    plt.savefig(path + '/' + 'loss' + '_' + str(k - 1) + '.jpg')
    plt.close()
    torch.save(model.module,'tfn_trained_model.pth')
    print('moxingyibaocun')
    return model, optimizer, metric_logger


def test(opt, model, data, split, device):
    model.eval()

    custom_data_loader = PathomicFastDatasetLoader(opt, data, split,
                                                        mode=opt.mode) if opt.use_vgg_features else PathomicDatasetLoader(
        opt, data, split=split, mode=opt.mode)
    test_loader = torch.utils.data.DataLoader(dataset=custom_data_loader, batch_size=opt.batch_size, shuffle=False,
                                              collate_fn=mixed_collate, drop_last=False)
    risk_pred_all, censor_all, survtime_all = np.array([]), np.array([]), np.array([])
    probs_all, gt_all = None, np.array([])
    loss_test, grad_acc_test = 0, 0
    with torch.no_grad():
        for batch_idx, (x_path, x_grph, x_omic, censor, survtime, grade) in enumerate(test_loader):
            censor = censor.to(device) if "surv" in opt.task else censor
            grade = grade.to(device) if "grad" in opt.task else grade

            _, pred = model(x_path=x_path.to(device), x_grph=x_grph.to(device), x_omic=x_omic.to(device))

            loss_cox = CoxLoss(survtime, censor, pred, device) if opt.task == "surv" else 0
            loss_reg = define_reg(opt, model)
            loss_nll = F.nll_loss(pred, grade) if opt.task == "grad" else 0
            loss = opt.lambda_cox * loss_cox + opt.lambda_nll * loss_nll + opt.lambda_reg * loss_reg
            loss_test += loss.data.item()

            gt_all = np.concatenate((gt_all, grade.detach().cpu().numpy().reshape(-1)))  # Logging Information

            if opt.task == "surv":
                risk_pred_all = np.concatenate(
                    (risk_pred_all, pred.detach().cpu().numpy().reshape(-1)))  # Logging Information
                censor_all = np.concatenate((censor_all, censor.detach().cpu().numpy().reshape(-1)))  # Logging Information
                survtime_all = np.concatenate(
                    (survtime_all, survtime.detach().cpu().numpy().reshape(-1)))  # Logging Information

    ################################################### 
    # ==== Measuring Test Loss, C-Index, P-Value ==== #
    ###################################################
    loss_test /= len(test_loader)
    cindex_test = CIndex_lifeline(risk_pred_all, censor_all, survtime_all) if opt.task == 'surv' else None
    pvalue_test = cox_log_rank(risk_pred_all, censor_all, survtime_all) if opt.task == 'surv' else None
    surv_acc_test = accuracy_cox(risk_pred_all, censor_all) if opt.task == 'surv' else None
    grad_acc_test = None
    pred_test = [risk_pred_all, survtime_all, censor_all, probs_all, gt_all]
    torch.save(model,'lmf_test_model.pth')

    return loss_test, cindex_test, pvalue_test, surv_acc_test, grad_acc_test, pred_test
