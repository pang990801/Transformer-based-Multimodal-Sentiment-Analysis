import torch
import os
import sys
import csv
from tqdm import tqdm

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

from training_structures.Supervised_Learning import train, test  # noqa
from models import ConcatLate, ConcatEarly, LowRankTensorFusion, TensorFusion, EarlyFusionTransformer, \
    LateFusionTransformer  # noqa
from data.get_dataloader import get_dataloader, get_mosi_dataloader  # noqa


def run_experiment(fusion_method, testdata):
    """Run a single experiment for a specific fusion method."""
    results = []


    # test the model
    model_save_path = f"../checkpoints/glove_{fusion_method}.pt"
    model = torch.load(model_save_path).cuda()
    test_results = test(
        model=model, test_dataloaders_all=testdata, dataset='mosi',
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

glove_filepath = '../data/MOSI/mosi_raw_glove.pkl'
batch_size = 32

testdata_OT = get_mosi_dataloader(
    glove_filepath, batch_size=batch_size, data_type='mosi', num_workers=0, robust_test=False
)

testdata_TE = get_mosi_dataloader(
    glove_filepath, batch_size=batch_size, max_pad=True, data_type='mosi', num_workers=0, robust_test=False
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
    with open('mosi_glove_results.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Fusion Method', 'MAE', 'ACC7', 'Acc5', 'ACC2', 'Corr', 'F1'])
        writer.writerows(all_results)

    print("All experiments completed. Results saved to 'mosi_glove_results.csv'.")
