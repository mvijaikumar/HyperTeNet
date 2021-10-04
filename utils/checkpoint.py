import os
import sys
import torch


def save_model(state, fname):
    print("Saving model state...", end="")
    torch.save(state, fname)
    print("\rCheckpoint saved!")


def load_model(model, fname, device):
    print("Loading model...", end="")
    if os.path.exists(fname):
        model.load_state_dict(torch.load(fname, map_location=device))
        print("\rModel loaded successfully!")
    else:
        print("\rNo saved model found! Exiting.")
        sys.exit()
    return model
