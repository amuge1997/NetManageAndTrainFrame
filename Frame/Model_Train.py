import torch.nn as nn
import torch.optim as optim
import time


class Train:

    def __init__(self):
        pass

    def train(self, model, loader, dc_train_params,):

        lr = dc_train_params['lr']
        epochs = dc_train_params['epochs']
        lossf_sel = dc_train_params['lossf']
        opt_sel = dc_train_params['optim']
        momentum = dc_train_params['momentum']
        is_show_details = dc_train_params['is_show_details']


        if lossf_sel == 'mse':
            loss_func = nn.MSELoss()
        elif lossf_sel == 'smo':
            loss_func = nn.SmoothL1Loss()
        elif lossf_sel == 'bce':
            loss_func = nn.BCELoss
        else:
            raise Exception('loss function')

        if opt_sel == 'adam':
            optimizer = optim.Adam(model.parameters(),lr=lr)
        elif opt_sel == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
        else:
            raise Exception('optimizer function')


        dc_model = {
            'model':model,
            'train_log':None,
        }

        train_log = {
            'lr':lr,
            'epochs':epochs,
            'lossf':lossf_sel,
            'optim':opt_sel,
            'momentum':momentum,
            'loss':None
        }

        # 模型训练
        ls_loss = []
        ls_rate = []

        start = time.time()
        for epoch in range(epochs):
            print()
            train_lossSum = 0
            for step, (batch_x, batch_y) in enumerate(loader):
                # 正向计算获得输出
                Y = model(batch_x)
                # 与模型连接
                loss = loss_func(Y, batch_y)
                # 梯度初始化归零,准备优化
                optimizer.zero_grad()
                # 反向传播,更新梯度
                loss.backward()
                # 根据计算得到的梯度,结合优化器参数进行模型参数更新
                optimizer.step()
                train_loss = loss.item()
                train_lossSum += train_loss
                if is_show_details:
                    print('{}-{}: {}'.format(epoch, step, train_loss))
            train_lossSum = train_lossSum / len(loader)
            print()
            print('{}-mean: {}'.format(epoch, train_lossSum))
            ls_loss.append(train_lossSum)

            if len(ls_loss) > 1:
                fl_rateFirst = ls_loss[-1] / ls_loss[0]
                fl_rateLast = ls_loss[-1] / ls_loss[-2]
                ls_rate.append(fl_rateFirst)
                print('{}-rate-compare with first: {}'.format(epoch, fl_rateFirst))
                print('{}-rate-compare with last: {}'.format(epoch, fl_rateLast))
            print()
        end = time.time()

        use_time = end - start

        train_log['time'] = use_time
        train_log['loss'] = ls_loss
        dc_model['train_log'] = train_log

        return dc_model














