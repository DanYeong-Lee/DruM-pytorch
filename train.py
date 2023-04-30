from tqdm import tqdm
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.datasets.qm9_dataset import QM9Dataset
from src.data_utils import collate_fn, drop_masked, data_to_mol, mol_to_pil, valid_mol_to_pil, get_n_atom_distribution
from src.model import GraphTransformer
from src.diffusion import SDE

from torch_ema import ExponentialMovingAverage

import os
import wandb
from rdkit.Chem.rdmolops import SanitizeMol


def train(model, sde, optimizer, ema, dataloader, device):
    model.train()
    epoch_loss = 0.
    for i, (X, E, mask) in enumerate(tqdm(dataloader)):
        optimizer.zero_grad()
        X, E, mask = X.to(device), E.to(device), mask.to(device)
        loss = sde.training_loss(model, X, E, mask)
        loss.backward()
        optimizer.step()
        ema.update()
        epoch_loss += loss.item()
    return epoch_loss / len(dataloader)



def main():
    wandb.init(project='drum', name='drum-qm9-test')
    df = pd.read_csv('../../data/qm9.csv')
    dataset = QM9Dataset(df['smiles'].tolist())
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True, collate_fn=collate_fn, num_workers=4, drop_last=True)

    device = torch.device('cuda:1')

    n_layers = 6
    input_dims = {'X': 4, 'E': 1, 'y': 1}
    hidden_mlp_dims = {'X': 256, 'E': 128, 'y': 128}
    hidden_dims = {'dx': 256, 'de': 64, 'dy': 64, 'n_head': 8, 'dim_ffX': 256, 'dim_ffE': 128, 'dim_ffy': 128}
    output_dims = {'X': 4, 'E': 1, 'y': 1}

    model = GraphTransformer(
            n_layers=n_layers,
            input_dims=input_dims,
            hidden_mlp_dims=hidden_mlp_dims,
            hidden_dims=hidden_dims,
            output_dims=output_dims,
            act_fn_in=nn.ReLU(),
            act_fn_out=nn.ReLU(),
        ).to(device)
    sde = SDE().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-12)
    ema = ExponentialMovingAverage(model.parameters(), decay=0.999)


    n_samples = 100
    if os.path.exists('data/qm9_train_dist.pt'):
        train_data_dist = torch.load('data/qm9_train_dist.pt')
    else:
        train_data_dist = get_n_atom_distribution(dataset)
        torch.save(train_data_dist, 'data/qm9_train_dist.pt')

    for epoch in range(100):
        train_loss = train(model, sde, optimizer, ema, dataloader, device)
        with ema.average_parameters():
            model.eval()
            n_atoms = torch.multinomial(train_data_dist, n_samples, replacement=True)
            Xs, Es = sde.predictor_corrector_sample(model, device, n_atoms=n_atoms, n_steps=1000, n_lang_steps=1)

        valid_count = 0
        imgs = []
        for i in range(len(Xs)):
            x, e = drop_masked(Xs[i], Es[i])
            mol = data_to_mol(x, e)

            try:
                SanitizeMol(mol)
                pil = valid_mol_to_pil(mol)
                valid_count += 1
            except:
                pil = mol_to_pil(mol)

            imgs.append(wandb.Image(pil))
        
        wandb.log({
            'train_loss': train_loss,
            'valid_ratio': valid_count / n_samples, 
            'samples': imgs
            })

    with ema.average_parameters():
        torch.save(model.state_dict(), 'ckpts/qm9_test_100epochs.pt')


if __name__ == '__main__':
    main()