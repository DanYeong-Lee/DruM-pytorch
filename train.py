
from tqdm import tqdm
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.data_utils import collate_fn, drop_masked, data_to_mol, mol_to_pil, valid_mol_to_pil, get_n_atom_distribution
from src.model import GraphTransformer
from src.diffusion import DruM_SDE

from torch_ema import ExponentialMovingAverage

import os
import os.path as osp
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


def validation(model, sde, ema, train_data_dist, n_samples, device):
    model.eval()
    with ema.average_parameters():
        model.eval()
        n_atoms = torch.multinomial(train_data_dist, n_samples, replacement=True)
        Xs, Es = sde.predictor_corrector_sample(model, device, n_atoms=n_atoms, n_steps=1000, n_lang_steps=1)

    valid_count = 0.
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

    valid_ratio = valid_count / n_samples

    return valid_ratio, imgs


def main(args):
    name = f'DruM_{args.dataset}_layers{args.n_layers}_adj-orders{args.n_orders}_bsz{args.bsz}_epochs{args.epochs}'
    wandb.init(project='drum', name=name, config=args)

    if args.dataset == 'qm9':
        from src.datasets.qm9_dataset import QM9Dataset
        df = pd.read_csv('data/qm9.csv')
        dataset = QM9Dataset(df['smiles'].tolist())
    elif args.dataset == 'zinc':
        from src.datasets.zinc_dataset import ZINCDataset
        df = pd.read_csv('data/zinc_250k.csv')
        dataset = ZINCDataset(df)
    else:
        raise NotImplementedError

    dataloader = DataLoader(dataset, batch_size=args.bsz, shuffle=True, collate_fn=collate_fn, num_workers=4, drop_last=True)

    device = torch.device(args.device)

    n_layers = args.n_layers
    x_dim = 4 if args.dataset == 'qm9' else 9

    input_dims = {'X': x_dim, 'E': args.n_orders, 'y': 1}
    hidden_mlp_dims = {'X': 256, 'E': 128, 'y': 128}
    hidden_dims = {'dx': 256, 'de': 64, 'dy': 64, 'n_head': 8, 'dim_ffX': 256, 'dim_ffE': 128, 'dim_ffy': 128}
    output_dims = {'X': x_dim, 'E': 1, 'y': 1}

    model = GraphTransformer(
            n_layers=n_layers,
            input_dims=input_dims,
            hidden_mlp_dims=hidden_mlp_dims,
            hidden_dims=hidden_dims,
            output_dims=output_dims,
            act_fn_in=nn.ReLU(),
            act_fn_out=nn.ReLU(),
        ).to(device)
    sde = DruM_SDE().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-12)
    ema = ExponentialMovingAverage(model.parameters(), decay=0.999)

    n_epochs = args.epochs
    n_samples = 100
    validation_interval = 50

    if osp.exists(f'data/{args.dataset}_train_dist.pt'):
        train_data_dist = torch.load(f'data/{args.dataset}_train_dist.pt')
    else:
        train_data_dist = get_n_atom_distribution(dataset)
        torch.save(train_data_dist, f'data/{args.dataset}_train_dist.pt')

    for epoch in range(n_epochs):
        train_loss = train(model, sde, optimizer, ema, dataloader, device)
        
        if epoch % validation_interval == 0:
            valid_ratio, imgs = validation(model, sde, ema, train_data_dist, n_samples, device)
            wandb.log({
                'train_loss': train_loss,
                'valid_ratio': valid_ratio, 
                'samples': imgs
                })
        else:
            wandb.log({
                'train_loss': train_loss,
                })

    os.makedirs('ckpts', exist_ok=True)
    with ema.average_parameters():
        torch.save(model.state_dict(), f'ckpts/{name}.pt')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--bsz', type=int, default=256)
    parser.add_argument('--n_layers', type=int, default=6)
    parser.add_argument('--n_orders', type=int, default=1)
    parser.add_argument('--dataset', type=str, default='qm9')
    parser.add_argument('--device', type=str, default='cuda:1')
    args = parser.parse_args()
    main(args)