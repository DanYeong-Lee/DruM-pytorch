import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch_geometric
from torch_geometric.data import Data
from rdkit import Chem, RDLogger
from src.data_utils import collate_fn



atom_type = {'C': 0, 'N': 1, 'O': 2, 'F': 3, 'S': 4, 'P': 5, 'Cl': 6, 'Br': 7, 'I': 8}
bond_type = {'SINGLE': 1, 'DOUBLE': 2, 'TRIPLE': 3}


def from_smiles(smiles: str) -> 'torch_geometric.data.Data':
    RDLogger.DisableLog('rdApp.*')

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        mol = Chem.MolFromSmiles('')
    Chem.Kekulize(mol)

    x = []
    for atom in mol.GetAtoms():
        x.append(atom_type[atom.GetSymbol()])
    x = torch.tensor(x, dtype=torch.long)
    x = F.one_hot(x, num_classes=len(atom_type)).to(torch.float)

    adj = torch.zeros((mol.GetNumAtoms(), mol.GetNumAtoms()), dtype=torch.float)

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        e = bond_type[str(bond.GetBondType())]
        adj[i, j] = e
        adj[j, i] = e

    return Data(x=x, adj=adj, smiles=smiles)



class ZINCDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        data = from_smiles(row['smiles'])
        data.adj = data.adj / 3
        data.aromatic_rings = torch.tensor(row['aromatic_rings'], dtype=torch.long)
        data.aliphatic_rings = torch.tensor(row['aliphatic_rings'], dtype=torch.long)
        data.hb_donors = torch.tensor(row['hb_donor_count'], dtype=torch.long)
        data.hb_acceptors = torch.tensor(row['hb_acceptor_count'], dtype=torch.long)

        return data