import datetime
import os
import warnings

import numpy as np
import torch

from data.embed_dataset import EmbedDataset
from models.models_base import Models
from utils.arguments import parse_args
from utils.parameters import Parameters
from utils.checkpoint import load_model
from utils.valid_test_error_seq import ValidTestErrorSEQ

warnings.filterwarnings("ignore", category=DeprecationWarning)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    args = parse_args()
    print('Loading dataset...', end="")
    args.device = device
    args.date_time = datetime.datetime.now()

    dataset = EmbedDataset(args)
    params = Parameters(args, dataset)

    print("\rDataset Statistics:")
    print(f"    Users: {params.num_user} | Lists: {params.num_list} | Items:{params.num_item}")
    print(f"    Train: {params.num_train_instances} | Valid: {params.num_valid_instances} | Test: {params.num_test_instances}")
    print(f"    Density: {100 * params.num_train_instances / (params.num_list * params.num_item):.4f} %")

    save_model_path = os.path.join("./saved_models", params.dataset + ".pth.tar")

    models = Models(params, device=device)
    model = models.get_model()
    model = load_model(model, save_model_path, device)
    model.to(device)

    vt_err = ValidTestErrorSEQ(params)

    print("=" * 60)
    print("Calculating test errors...", end="")
    (test_hits_lst, test_ndcg_lst, test_map_lst) = vt_err.get_update(model, 0, device, valid_flag=False)
    (test_hr, test_ndcg, test_map) = (np.mean(test_hits_lst), np.mean(test_ndcg_lst), np.mean(test_map_lst))
    print(f"\rTest HR: {test_hr:.4f} | Test MAP: {test_map:.4f} | Test NDCG: {test_ndcg:.4f}")
    print("=" * 60)
