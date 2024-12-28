
"""Implements supervised learning training procedures."""
from sklearn.metrics import accuracy_score, f1_score
import torch
from torch import nn
import time
from tqdm import tqdm
import numpy as np
#import pdb
from scipy.stats import pearsonr
softmax = nn.Softmax()



def eval_affect(truths, results, exclude_zero=True):
    if type(results) is np.ndarray:
        test_preds = results
        test_truth = truths
    else:
        test_preds = results.cpu().numpy()
        test_truth = truths.cpu().numpy()

    non_zeros = np.array([i for i, e in enumerate(
        test_truth) if e != 0 or (not exclude_zero)])

    binary_truth = (test_truth[non_zeros] > 0)
    binary_preds = (test_preds[non_zeros] > 0)

    # 计算 F1 分数
    f1 = f1_score(binary_truth, binary_preds, average='binary')
    # 计算准确率
    accuracy = accuracy_score(binary_truth, binary_preds)

    return f1, accuracy

class MultiFramework(nn.Module):
    """Implements MMDL classifier."""
    
    def __init__(self, encoders, fusion, head, has_padding=False):

        super(MultiFramework, self).__init__()
        self.encoders = nn.ModuleList(encoders)
        self.fuse = fusion
        self.head = head
        self.has_padding = has_padding
        self.fuseout = None
        self.reps = []

    def forward(self, inputs):
        num_encoders = len(self.encoders)
        outs = []

        # 处理输入并通过编码器获取输出
        if self.has_padding:
            # 如果有填充，输入为两部分：数据和填充信息
            for i in range(num_encoders):
                outs.append(self.encoders[i]([inputs[0][i], inputs[1][i]]))
        else:
            # 没有填充的情况，直接处理输入
            for i in range(num_encoders):
                outs.append(self.encoders[i](inputs[i]))

        # 存储编码器输出
        self.reps = outs

        # 根据输出类型选择如何融合
        if self.has_padding and not isinstance(outs[0], torch.Tensor):
            # 如果有填充，并且输出不是张量（可能是元组），仅使用每个输出的第一部分
            outs = [out[0] for out in outs]
        out = self.fuse(outs)

        # 存储融合后的输出
        self.fuseout = out

        # 处理融合后的输出，如果是元组则取第一元素
        if isinstance(out, tuple):
            out = out[0]

        # 如果有填充并且原始输出不是张量，需要处理额外的填充信息
        if self.has_padding and not isinstance(self.reps[0], torch.Tensor):
            return self.head([out, inputs[1][0]])

        # 返回最终模型头部的输出
        return self.head(out)





def deal_with_objective(objective, pred, truth, args):
    """Alter inputs depending on objective function, to deal with different objective arguments."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if type(objective) == nn.CrossEntropyLoss:
        if len(truth.size()) == len(pred.size()):
            truth1 = truth.squeeze(len(pred.size())-1)
        else:
            truth1 = truth
        return objective(pred, truth1.long().to(device))
    elif type(objective) == nn.MSELoss or type(objective) == nn.modules.loss.BCEWithLogitsLoss or type(objective) == nn.L1Loss:
        return objective(pred, truth.float().to(device))
    else:
        return objective(pred, truth, args)


from memory_profiler import memory_usage
import time


def getallparams(li):
    params = 0
    for module in li:
        for param in module.parameters():
            params += param.numel()
    return params


def all_in_one_train(trainprocess, trainmodules):
    starttime = time.time()
    mem = max(memory_usage(proc=trainprocess))
    endtime = time.time()

    print("Training Time: "+str(endtime-starttime))
    print("Training Peak Mem: "+str(mem))
    print("Training Params: "+str(getallparams(trainmodules)))


def all_in_one_test(testprocess, testmodules):
    teststart = time.time()
    testprocess()
    testend = time.time()
    print("Inference Time: "+str(testend-teststart))
    print("Inference Params: "+str(getallparams(testmodules)))




def train(
        encoders, fusion, head, train_dataloader, valid_dataloader, total_epochs, additional_optimizing_modules=[], is_packed=False,
        early_stop=False, optimtype=torch.optim.RMSprop, lr=0.001, weight_decay=0.0,
        objective=nn.CrossEntropyLoss(), save='best.pt', validtime=False, objective_args_dict=None, input_to_float=True, clip_val=8,
        track_complexity=True):
    """
    Handle running a simple supervised training loop.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MultiFramework(encoders, fusion, head, has_padding=is_packed).to(device)
    def _trainprocess():
        additional_params = []
        for m in additional_optimizing_modules:
            additional_params.extend(
                [p for p in m.parameters() if p.requires_grad])
        op = optimtype([p for p in model.parameters() if p.requires_grad] +
                       additional_params, lr=lr, weight_decay=weight_decay)
        bestvalloss = float('inf')
        patience = 0

        def _processinput(inp):
            if input_to_float:
                return inp.float()
            else:
                return inp

        for epoch in range(total_epochs):
            totalloss = 0.0
            totals = 0
            model.train()

            # 添加训练阶段的进度条
            train_iter = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{total_epochs} Training")

            for j in train_iter:
                op.zero_grad()
                if is_packed:
                    with torch.backends.cudnn.flags(enabled=False):
                        out = model([[_processinput(i).to(device)
                                      for i in j[0]], j[1]])
                else:
                    out = model([_processinput(i).to(device)
                                 for i in j[:-1]])

                if objective_args_dict is not None:
                    objective_args_dict['reps'] = model.reps
                    objective_args_dict['fused'] = model.fuseout
                    objective_args_dict['inputs'] = j[:-1]
                    objective_args_dict['training'] = True
                    objective_args_dict['model'] = model

                loss = deal_with_objective(objective, out, j[-1], objective_args_dict)
                totalloss += loss.item() * len(j[-1])
                totals += len(j[-1])

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val)
                op.step()

            print(f"Epoch {epoch+1} train loss: {totalloss / totals}")

            model.eval()
            with torch.no_grad():
                totalloss = 0.0
                true = []
                pts = []

                # 添加验证阶段的进度条
                valid_iter = tqdm(valid_dataloader, desc=f"Epoch {epoch+1}/{total_epochs} Validation")

                for j in valid_iter:
                    if is_packed:
                        out = model([[_processinput(i).to(device)
                                      for i in j[0]], j[1]])
                    else:
                        out = model([_processinput(i).to(device)
                                     for i in j[:-1]])

                    if objective_args_dict is not None:
                        objective_args_dict['reps'] = model.reps
                        objective_args_dict['fused'] = model.fuseout
                        objective_args_dict['inputs'] = j[:-1]
                        objective_args_dict['training'] = False

                    loss = deal_with_objective(objective, out, j[-1], objective_args_dict)
                    totalloss += loss.item() * len(j[-1])

                    true.append(j[-1])


            true = torch.cat(true, 0)
            totals = true.shape[0]
            valloss = totalloss / totals
            print(f"Epoch {epoch+1} valid loss: {valloss:.4f}")

            if valloss < bestvalloss:
                patience = 0
                bestvalloss = valloss
                print("Saving Best Model")
                torch.save(model, save)
            else:
                patience += 1

            if early_stop and patience > 7:
                print("Early stopping due to no improvement in validation loss.")
                break



    if track_complexity:
        all_in_one_train(_trainprocess, [model] + additional_optimizing_modules)
    else:
        _trainprocess()

def split_uniform_7(data):
    """
    将 [-3, 3] 等分成 7 段，返回每个元素对应的区间索引(1..7)。
    """
    # 计算步长
    step = 6.0 / 7.0  # 6.0 = 3 - (-3)
    # 定义边界 edges, 共有 7 段，需要 8 个边界
    edges = [-3.0 + i*step for i in range(8)]  # i=0..7
    # 利用 np.digitize，把 data 中的值映射到 1..7
    # np.digitize: 给出 data 中每个值在 edges 中所处的位置(1-based)
    categories = np.digitize(data, edges, right=False)
    # 上面得到的类别范围 1..8，但超过 7 的部分说明 data==3.0 或更大，需要裁到 7
    categories = np.clip(categories, 1, 7)
    return categories

def split_uniform_5(data):
    """
    将 [-3, 3] 等分成 5 段，返回每个元素对应的区间索引(1..5)。
    """
    step = 6.0 / 5.0
    edges = [-3.0 + i*step for i in range(6)]  # i=0..5
    categories = np.digitize(data, edges, right=False)
    categories = np.clip(categories, 1, 5)
    return categories


def single_test(
        model, test_dataloader, is_packed=False,
        criterion=torch.nn.CrossEntropyLoss(), input_to_float=True):
    def _processinput(inp):
        return inp.float() if input_to_float else inp

    with torch.no_grad():
        totalloss = 0.0
        pred = []
        true = []
        pts = []
        all_oute = []
        for j in test_dataloader:
            model.eval()
            if is_packed:
                out = model([[_processinput(i).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
                            for i in j[0]], j[1]])
            else:
                out = model([_processinput(i).float().to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
                            for i in j[:-1]])
            if type(criterion) == torch.nn.modules.loss.BCEWithLogitsLoss or type(criterion) == torch.nn.MSELoss:
                loss = criterion(out, j[-1].float().to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")))
            elif type(criterion) == nn.CrossEntropyLoss:
                if len(j[-1].size()) == len(out.size()):
                    truth1 = j[-1].squeeze(len(out.size())-1)
                else:
                    truth1 = j[-1]
                loss = criterion(out, truth1.long().to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")))
            else:
                loss = criterion(out, j[-1].to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")))
            totalloss += loss*len(j[-1])
            prede = []
            oute = out.cpu().numpy().tolist() #预测的结果
            all_oute.extend(oute)
            for i in oute:   #转换成1，-1，0
                if i[0] > 0:
                    prede.append(1)
                elif i[0] < 0:
                    prede.append(-1)
                else:
                    prede.append(0)
            pred.append(torch.LongTensor(prede))
            true.append(j[-1])

        if oute:
            pred_reg = torch.Tensor(all_oute)  #回归预测值
        if pred:
            pred = torch.cat(pred, 0)  #分类预测结果
        true = torch.cat(true, 0)
        totals = true.shape[0]
        testloss = totalloss/totals

        true_vals = true  #真实值


        # 1) 计算回归指标: MSE, MAE, Corr
        corr, _ = pearsonr(true_vals.squeeze(), pred_reg.squeeze())
        mse = torch.mean((true_vals - pred_reg) ** 2).item()
        mae = torch.mean(torch.abs(true_vals - pred_reg)).item()

        # 2) 计算分类指标: Acc7 (均匀区间), Acc5 (均匀区间)
        #    先把回归输出和真实值都映射到1..7 / 1..5
        pred_7 = split_uniform_7(pred_reg)
        true_7 = split_uniform_7(true_vals)
        Acc7 = accuracy_score(true_7, pred_7)

        pred_5 = split_uniform_5(pred_reg)
        true_5 = split_uniform_5(true_vals)
        Acc5 = accuracy_score(true_5, pred_5)


        F1, Acc2 = eval_affect(true_vals, pred_reg)

        import numpy as np
        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

        conf_matrix = confusion_matrix(pred_7, true_7)

        disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=np.unique(true_7))
        disp.plot(cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.show()

        # 输出
        print(f"Test Loss: {testloss:.4f}")
        print(f"MSE: {mse:.4f}, MAE: {mae:.4f}, Corr: {corr:.4f}")
        print(f"Acc7 (uniform splits): {Acc7:.4f}")
        print(f"Acc5 (uniform splits): {Acc5:.4f}")
        print(f"Acc2: {Acc2:.4f}")
        print(f"F1 score: {F1:.4f}")

        return {
            'TestLoss': testloss,
            'MSE': mse,
            'MAE': mae,
            'Corr': corr,
            'Acc7_uniform': Acc7,
            'Acc5_uniform': Acc5,
            'Acc2': Acc2,
            'F1': F1
        }


def test(
        model, test_dataloaders_all, is_packed=False, criterion=nn.CrossEntropyLoss(), input_to_float=True):
    """
    Handle getting test results for a simple supervised training loop.
    
    :param model: saved checkpoint filename from train
    :param test_dataloaders_all: test data
    :param dataset: the name of dataset, need to be set for testing effective robustness
    :param criterion: only needed for regression, put MSELoss there   
    """
    def _testprocess():
        results = single_test(model, test_dataloaders_all, is_packed, criterion, input_to_float)
        return results  # 返回结果
    all_in_one_test(_testprocess, [model])
    results = _testprocess()
    return results  # 直接返回 test 结果



