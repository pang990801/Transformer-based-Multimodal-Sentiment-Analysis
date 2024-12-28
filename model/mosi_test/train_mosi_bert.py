import torch
import os
import sys
import csv
from tqdm import tqdm

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

from train_and_test import train, test
from models import ConcatLate, ConcatEarly, LowRankTensorFusion, TensorFusion, EarlyFusionTransformer, \
    LateFusionTransformer  ,Identity, MLP, LSTM, GRUWithLinear, GRU, TransformerSeq

from data.get_dataloader import get_dataloader, get_mosi_dataloader


def run_experiment(fusion_method, testdata):
    """Run a single experiment for a specific fusion method."""
    input_dims = [35, 74, 768]
    results = []

    # Define encoders, fusion, and head based on fusion_method
    if fusion_method == 'ConcatEarly':
        encoders = [Identity().cuda() for _ in input_dims]
        head = torch.nn.Sequential(
            LSTM(877, 1024, dropout=True, has_padding=True),
            MLP(1024, 1024, 1)
        ).cuda()
        fusion = ConcatEarly().cuda()
    elif fusion_method == 'Concat':
        encoders = [
            LSTM(35, 64, dropout=True, has_padding=True).cuda(),
            LSTM(74, 256, dropout=True, has_padding=True).cuda(),
            LSTM(768, 1024, dropout=True, has_padding=True).cuda()
        ]
        head = MLP(1344, 1344, 1).cuda()
        fusion = ConcatLate().cuda()
    elif fusion_method == 'LowRankTensorFusion':
        encoders = [
            GRUWithLinear(35, 64, 32, dropout=True, has_padding=True).cuda(),
            GRUWithLinear(74, 256, 64, dropout=True, has_padding=True).cuda(),
            GRUWithLinear(768, 1024, 256, dropout=True, has_padding=True).cuda()
        ]
        head = MLP(256, 256, 1).cuda()
        fusion = LowRankTensorFusion([32, 64, 256], 256, 32).cuda()
    elif fusion_method == 'TensorFusion':
        encoders = [
            GRUWithLinear(35, 64, 19, dropout=True, has_padding=True).cuda(),
            GRUWithLinear(74, 256, 39, dropout=True, has_padding=True).cuda(),
            GRUWithLinear(768, 1024, 159, dropout=True, has_padding=True).cuda()
        ]
        head = MLP(128000, 2048, 1).cuda()
        fusion = TensorFusion().cuda()
    elif fusion_method == 'TransformerEarly':
        encoders = [Identity().cuda() for _ in input_dims]
        head = MLP(64, 64, 1).cuda()
        fusion = EarlyFusionTransformer(n_features=877).cuda()
    elif fusion_method == 'TransformerLate':
        encoders = [TransformerSeq(35, 64).cuda(),
                    TransformerSeq(74, 128).cuda(),
                    TransformerSeq(768, 1024).cuda()]
        head = MLP(32, 32, 1).cuda()
        fusion = LateFusionTransformer(in_dim=1216).cuda()
    else:
        raise ValueError(f"Invalid fusion method: {fusion_method}")

    # test the model
    model_save_path = f"../checkpoints/{fusion_method}.pt"
    model = torch.load(model_save_path).cuda()
    test_results = test(
        model=model, test_dataloaders_all=testdata, dataset='mosi',
        is_packed=False if fusion_method == 'TransformerEarly' else True, criterion=torch.nn.L1Loss()
    )

    # Extract results
    mae = round(test_results["MAE"], 4)
    acc7 = round(test_results["Acc7_uniform"], 4)
    acc5 = round(test_results["Acc5_uniform"], 4)
    acc2 = round(test_results["Acc2"], 4)
    corr = round(test_results["Corr"], 4)
    f1 = round(test_results["F1"], 4)

    results.append([fusion_method, mae, acc7, acc5, acc2, corr, f1])
    return results

bert_filepath = '../data/MOSI/mosi_raw_bert.pkl'
batch_size = 32

testdata_OT = get_mosi_dataloader(
    bert_filepath, batch_size=batch_size, data_type='mosi', num_workers=0
)
testdata_TE = get_mosi_dataloader(
    bert_filepath, batch_size=batch_size, max_pad=True, data_type='mosi', num_workers=0
)
if __name__ == '__main__':



    fusion_methods = [
        'ConcatEarly',
        'Concat',
        'LowRankTensorFusion',
        'TensorFusion',
        'TransformerEarly',
        'TransformerLate'
    ]
    all_results = []

    for fusion in fusion_methods:
        print(fusion)
        if fusion == 'TransformerEarly':
            testdata = testdata_TE
        else:
            testdata = testdata_OT
        results = run_experiment(fusion, testdata)
        all_results.extend(results)

    # Save results to CSV
    with open('mosi_bert_results.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Fusion Method', 'MAE', 'ACC7', 'Acc5', 'ACC2', 'Corr', 'F1'])
        writer.writerows(all_results)

    print("All experiments completed. Results saved to 'mosi_bert_results.csv'.")