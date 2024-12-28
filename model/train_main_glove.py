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

from data.get_dataloader import get_dataloader

def run_experiment(fusion_method, traindata, validdata, testdata):
    """Run a single experiment for a specific fusion method."""
    visual_dim = 35
    audio_dim = 74
    text_dim = 300

    input_dims = [visual_dim, audio_dim, text_dim]
    total_dim = visual_dim + audio_dim + text_dim
    results = []

    # Define encoders, fusion, and head based on fusion_method
    if fusion_method == 'ConcatEarly':
        encoders = [Identity().cuda() for _ in input_dims]
        head = torch.nn.Sequential(
            LSTM(total_dim, 512, dropout=True, has_padding=True),
            MLP(512, 512, 1)
        ).cuda()
        fusion = ConcatEarly().cuda()
    elif fusion_method == 'ConcatLate':
        encoders = [
            LSTM(visual_dim, 64, dropout=True, has_padding=True).cuda(),
            LSTM(audio_dim, 256, dropout=True, has_padding=True).cuda(),
            LSTM(text_dim, 512, dropout=True, has_padding=True).cuda()
        ]
        head = MLP(832, 832, 1).cuda()
        fusion = ConcatLate().cuda()
    elif fusion_method == 'LowRankTensorFusion':
        encoders = [
            GRUWithLinear(visual_dim, 64, 32, dropout=True, has_padding=True).cuda(),
            GRUWithLinear(audio_dim, 256, 64, dropout=True, has_padding=True).cuda(),
            GRUWithLinear(text_dim, 512, 128, dropout=True, has_padding=True).cuda()
        ]
        head = MLP(128, 128, 1).cuda()
        fusion = LowRankTensorFusion([32, 64, 128], 128, 32).cuda()
    elif fusion_method == 'TensorFusion':
        encoders = [
            GRUWithLinear(visual_dim, 64, 19, dropout=True, has_padding=True).cuda(),
            GRUWithLinear(audio_dim, 256, 39, dropout=True, has_padding=True).cuda(),
            GRUWithLinear(text_dim, 512, 79, dropout=True, has_padding=True).cuda()
        ]
        head = MLP(64000, 2048, 1).cuda()
        fusion = TensorFusion().cuda()
    elif fusion_method == 'TransformerEarly':
        encoders = [Identity().cuda() for _ in input_dims]
        #head = MLP(16, 16, 1).cuda()
        head = Identity().cuda()
        fusion = EarlyFusionTransformer(n_features=409).cuda()
    elif fusion_method == 'TransformerLate':
        encoders = [TransformerSeq(visual_dim, 64).cuda(),
                    TransformerSeq(audio_dim, 128).cuda(),
                    TransformerSeq(text_dim, 512).cuda()]
        head = MLP(32, 32, 1).cuda()
        fusion = LateFusionTransformer(in_dim=1792).cuda()
    else:
        raise ValueError(f"Invalid fusion method: {fusion_method}")

    # Train and test the model
    model_save_path = f"checkpoints/glove_{fusion_method}.pt"
    # train(
    #     encoders, fusion, head, traindata, validdata, 10,
    #     optimtype=torch.optim.AdamW, early_stop=True, is_packed=False if fusion_method == 'TransformerEarly' else True,
    #     lr=1e-4, save=model_save_path, weight_decay=0.01, objective=torch.nn.L1Loss()
    # )
    model = torch.load(model_save_path).cuda()
    test_results = test(
        model=model, test_dataloaders_all=testdata, dataset='mosei',
        is_packed=False if fusion_method == 'TransformerEarly' else True, criterion=torch.nn.L1Loss(), no_robust=True
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




if __name__ == '__main__':
    glove_filepath = 'data/MOSEI/mosei_raw_glove.pkl'
    batch_size = 32


    fusion_methods = [
        'ConcatEarly',
        'ConcatLate',
        'LowRankTensorFusion',
        'TensorFusion',
        'TransformerEarly',
        'TransformerLate'
    ]

    # fusion_methods = [
    # 'TransformerEarly', 'TransformerLate'
    # ]
    all_results = []

    for fusion in fusion_methods:
        print(fusion)
        if fusion == 'TransformerEarly':
            traindata, validdata, testdata = get_dataloader(
                glove_filepath, batch_size=batch_size, max_pad=True, data_type='mosei', num_workers=0, robust_test=False
            )
        else:
            traindata, validdata, testdata = get_dataloader(
                glove_filepath, batch_size=batch_size, data_type='mosei', num_workers=0, robust_test=False
            )
        results = run_experiment(fusion, traindata, validdata, testdata)
        all_results.extend(results)

    # Save results to CSV
    with open('glove_results.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Fusion Method', 'MAE', 'ACC7', 'Acc5', 'ACC2', 'Corr', 'F1'])
        writer.writerows(all_results)

    print("All experiments completed. Results saved to 'glove_results.csv'.")