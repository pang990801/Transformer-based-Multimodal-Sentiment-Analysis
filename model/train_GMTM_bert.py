import torch
import sys
import os


sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
import csv
from train_and_test import train, test
from models import GatedMultiTransfomerModel, Identity, MLP, TransformerSeq, Transformer, ConcatLate
from data.get_dataloader import get_dataloader, get_ablation_dataloader


class HParams:
    num_heads = 4
    layers = 4
    attn_dropout = 0.1
    attn_dropout_modalities = [0, 0, 0.1]
    relu_dropout = 0.1
    res_dropout = 0.1
    out_dropout = 0.1
    embed_dropout = 0.2
    embed_dim = 64
    attn_mask = True
    output_dim = 1
    all_steps = False
    modality_dropout = 0.2
    use_text_transformer = True


if __name__ == '__main__':
    filepath = 'data/MOSEI/mosei_raw_bert.pkl'
    batch_size = 32
    max_seq_len = 50

    # 定义消融实验模态组合
    modality_combinations = [
        # ['text'],
        # ['audio'],
        # ['visual'],
        # ['text', 'audio'],
        # ['text', 'visual'],
        # ['audio', 'visual'],
        ['text', 'audio', 'visual']
    ]

    results = []

    # 遍历模态组合
    for modalities in modality_combinations:
    #while True:
        #modalities = ['text', 'audio', 'visual']
        print(f"Training with modalities: {modalities}")

        # 加载完整数据
        traindata, validdata, testdata = get_ablation_dataloader(filepath, embedding='bert', batch_size=32, data_type='mosei', max_pad=True, num_workers=0,
                                                                 modalities=modalities)


        # 定义模型输入的维度
        input_dims = [35, 74, 768]

        # 定义模型结构
        print(input_dims)
        encoders = [Identity().cuda(), Identity().cuda(), Identity().cuda()]
        fusion = GatedMultiTransfomerModel(3, input_dims, hyp_params=HParams).cuda()
        head = Identity().cuda()

        # 训练
        # train(encoders, fusion, head, traindata, validdata, 50,
        #       optimtype=torch.optim.AdamW, early_stop=True, is_packed=False,
        #       lr=1e-4, save=f"checkpoints/ablation/model_{'+'.join(modalities)}.pt", weight_decay=0.01, objective=torch.nn.L1Loss())

        # 测试
        print("Testing...")
        model = torch.load(f"checkpoints/ablation/model_{'+'.join(modalities)}.pt").cuda()
        test_results = test(model=model, test_dataloaders_all=testdata, is_packed=False, criterion=torch.nn.L1Loss())

        # Extract results
        mae = round(test_results["MAE"], 4)
        acc7 = round(test_results["Acc7_uniform"], 4)
        acc5 = round(test_results["Acc5_uniform"], 4)
        acc2 = round(test_results["Acc2"], 4)
        corr = round(test_results["Corr"], 4)
        f1 = round(test_results["F1"], 4)

        results.append([modalities, mae, acc7, acc5, acc2, corr, f1])
        # 保存结果为表格

    # with open('ablation_results.csv', 'w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(['Fusion Method', 'MAE', 'ACC7', 'Acc5', 'ACC2', 'Corr', 'F1'])
    #     writer.writerows(results)
    #
    # print("All experiments completed. Results saved to ablation_results.csv.")