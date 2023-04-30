from tqdm import tqdm
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj
from torch_geometric.transforms import BaseTransform
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import MolToImage
from rdkit.Chem.rdchem import BondType as BT
    

def collate_fn(data_list):
    Xs, Es, masks = [], [], []
    n_atom_types = data_list[0].x.shape[1]

    n_max_nodes = max([data.num_nodes for data in data_list])
    for data in data_list:
        n_nodes = data.num_nodes
        X = torch.zeros((n_max_nodes, n_atom_types), dtype=torch.float)
        X[:n_nodes] = data.x
        Xs.append(X)

        E = torch.zeros(n_max_nodes, n_max_nodes, dtype=torch.float)
        E[:n_nodes, :n_nodes] = data.adj
        Es.append(E)

        mask = torch.tensor([1] * n_nodes + [0] * (n_max_nodes - n_nodes), dtype=torch.float)
        masks.append(mask)

    Xs = torch.stack(Xs)
    Es = torch.stack(Es)
    masks = torch.stack(masks)

    return Xs, Es, masks


def drop_masked(X, E):
    mask = (X.sum(-1) != 0)
    X = X[mask == 1]
    E = E[mask == 1]
    E = E[:, mask == 1]
    return X, E


def data_to_mol(X, E):
    atoms = ['C', 'N', 'O', 'F', 'S', 'P', 'Cl', 'Br', 'I']
    bonds = [BT.SINGLE, BT.DOUBLE, BT.TRIPLE]
    X = X.argmax(-1)
    E = torch.round(E * 3).long()  # 0: None, 1: Single, 2: Double, 3: Triple
    mol = Chem.RWMol()

    node_to_idx = {}
    for i, x in enumerate(X):
        atom = Chem.Atom(atoms[x.item()])
        mol_idx = mol.AddAtom(atom)
        node_to_idx[i] = mol_idx

    for ix, row in enumerate(E):
        for iy, bond in enumerate(row):
            if iy <= ix:
                continue
            if bond.item() == 0:
                continue
            
            bond_type = bonds[bond.item() - 1]
            mol.AddBond(node_to_idx[ix], node_to_idx[iy], bond_type)
    
    return mol.GetMol()


def mol_to_pil(mol):
    return MolToImage(mol, size=(300, 300))


def valid_mol_to_pil(mol):
    smiles = Chem.MolToSmiles(mol)
    mol = Chem.MolFromSmiles(smiles)
    AllChem.Compute2DCoords(mol)
    AllChem.GenerateDepictionMatching2DStructure(mol, mol)
    pil_image = MolToImage(mol, size=(300, 300))
    return pil_image


def get_n_atom_distribution(dataset):
    n_atom_distribution = torch.zeros(50, dtype=torch.long)
    for data in tqdm(dataset):
        n_atom_distribution[data.x.shape[0]] += 1
    n_atom_distribution = n_atom_distribution / n_atom_distribution.sum()

    return n_atom_distribution