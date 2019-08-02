import rdkit.Chem as Chem
from argparse import Namespace
import torch.nn.functional as F
from nn_utils import *
from typing import List, Tuple, Union
from typing import Dict, List, Union


ELEM_LIST=list(range(1,119))
ATOM_FDIM = len(ELEM_LIST) + 6 + 5 + 4 +5+1

BOND_FDIM = 5 + 6
MAX_NB = 6
SMILES_TO_GRAPH={}

def onek_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]

def atom_features(atom):

    return onek_encoding_unk(atom.GetAtomicNum() , ELEM_LIST) + onek_encoding_unk(atom.GetDegree(), [0, 1, 2, 3, 4, 5])+ onek_encoding_unk(atom.GetFormalCharge(), [-1, -2, 1, 2, 0]) + onek_encoding_unk(int(atom.GetChiralTag()), [0, 1, 2, 3])+onek_encoding_unk(int(atom.GetHybridization()),[
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2
    ])+[1 if atom.GetIsAromatic() else 0]

def bond_features(bond):
    bt = bond.GetBondType()
    stereo = int(bond.GetStereo())
    fbond = [bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE, bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC, bond.IsInRing()]
    fstereo = onek_encoding_unk(stereo, [0,1,2,3,4,5])
    fbond=fbond + fstereo
    return fbond


class MolGraph:
    """
    A MolGraph represents the graph structure and featurization of a single molecule.

    A MolGraph computes the following attributes:
    - smiles: Smiles string.
    - n_atoms: The number of atoms in the molecule.
    - n_bonds: The number of bonds in the molecule.
    - f_atoms: A mapping from an atom index to a list atom features.
    - f_bonds: A mapping from a bond index to a list of bond features.
    - a2b: A mapping from an atom index to a list of incoming bond indices.
    - b2a: A mapping from a bond index to the index of the atom the bond originates from.
    - b2revb: A mapping from a bond index to the index of the reverse bond.
    """

    def __init__(self, smiles: str, args: Namespace):
        """
        Computes the graph structure and featurization of a molecule.

        :param smiles: A smiles string.
        :param args: Arguments.
        """
        self.smiles = smiles
        self.n_atoms = 0
        self.n_bonds = 0
        self.f_atoms = []
        self.f_bonds = []
        self.a2b = []
        self.b2a = []
        self.b2revb = []


        mol = Chem.MolFromSmiles(smiles)


        self.n_atoms = mol.GetNumAtoms()


        for i, atom in enumerate(mol.GetAtoms()):
            self.f_atoms.append(atom_features(atom))
        self.f_atoms = [self.f_atoms[i] for i in range(self.n_atoms)]

        for _ in range(self.n_atoms):
            self.a2b.append([])


        for a1 in range(self.n_atoms):
            for a2 in range(a1 + 1, self.n_atoms):
                bond = mol.GetBondBetweenAtoms(a1, a2)
                if bond is None:
                    continue
                f_bond = bond_features(bond)
                if args.atom_messages:
                    self.f_bonds.append(f_bond)
                    self.f_bonds.append(f_bond)
                else:
                    self.f_bonds.append(self.f_atoms[a1] + f_bond)
                    self.f_bonds.append(self.f_atoms[a2] + f_bond)

                b1 = self.n_bonds
                b2 = b1 + 1

                self.a2b[a2].append(b1)
                self.b2a.append(a1)
                self.a2b[a1].append(b2)
                self.b2a.append(a2)
                self.b2revb.append(b2)
                self.b2revb.append(b1)
                self.n_bonds += 2

class BatchMolGraph:
    """
    A BatchMolGraph represents the graph structure and featurization of a batch of molecules.

    A BatchMolGraph contains the attributes of a MolGraph plus:
    - smiles_batch: A list of smiles strings.
    - n_mols: The number of molecules in the batch.
    - atom_fdim: The dimensionality of the atom features.
    - bond_fdim: The dimensionality of the bond features (technically the combined atom/bond features).
    - a_scope: A list of tuples indicating the start and end atom indices for each molecule.
    - b_scope: A list of tuples indicating the start and end bond indices for each molecule.
    - max_num_bonds: The maximum number of bonds neighboring an atom in this batch.
    - b2b: (Optional) A mapping from a bond index to incoming bond indices.
    - a2a: (Optional): A mapping from an atom index to neighboring atom indices.
    """

    def __init__(self, mol_graphs: List[MolGraph], args: Namespace):
        self.smiles_batch = [mol_graph.smiles for mol_graph in mol_graphs]
        self.n_mols = len(self.smiles_batch)

        self.atom_fdim = ATOM_FDIM
        self.bond_fdim = BOND_FDIM + (not args.atom_messages) * self.atom_fdim


        self.n_atoms = 1
        self.n_bonds = 1

        f_atoms = [[0] * self.atom_fdim]
        f_bonds = [[0] * self.bond_fdim]
        self.a_scope = []
        self.b_scope = []


        a2b = [[]]
        b2a = [0]
        b2revb = [0]
        for mol_graph in mol_graphs:
            f_atoms.extend(mol_graph.f_atoms)
            f_bonds.extend(mol_graph.f_bonds)

            for a in range(mol_graph.n_atoms):
                a2b.append([b + self.n_bonds for b in mol_graph.a2b[a]])

            for b in range(mol_graph.n_bonds):
                b2a.append(self.n_atoms + mol_graph.b2a[b])
                b2revb.append(self.n_bonds + mol_graph.b2revb[b])

            self.a_scope.append((self.n_atoms, mol_graph.n_atoms))
            self.b_scope.append((self.n_bonds, mol_graph.n_bonds))
            self.n_atoms += mol_graph.n_atoms
            self.n_bonds += mol_graph.n_bonds

        self.max_num_bonds = max(len(in_bonds) for in_bonds in a2b)

        self.f_atoms = torch.FloatTensor(f_atoms)
        self.f_bonds = torch.FloatTensor(f_bonds)
        self.a2b = torch.LongTensor([a2b[a] + [0] * (self.max_num_bonds - len(a2b[a])) for a in range(self.n_atoms)])

        self.b2a = torch.LongTensor(b2a)
        self.b2revb = torch.LongTensor(b2revb)
        self.b2b = None
        self.a2a = None


    def get_components(self) -> Tuple[torch.FloatTensor, torch.FloatTensor,
                                      torch.LongTensor, torch.LongTensor, torch.LongTensor,
                                      List[Tuple[int, int]], List[Tuple[int, int]]]:
        """
        Returns the components of the BatchMolGraph.

        :return: A tuple containing PyTorch tensors with the atom features, bond features, and graph structure
        and two lists indicating the scope of the atoms and bonds (i.e. which molecules they belong to).
        """
        return self.f_atoms, self.f_bonds, self.a2b, self.b2a, self.b2revb, self.a_scope, self.b_scope

    def get_b2b(self) -> torch.LongTensor:
        """
        Computes (if necessary) and returns a mapping from each bond index to all the incoming bond indices.

        :return: A PyTorch tensor containing the mapping from each bond index to all the incoming bond indices.
        """

        if self.b2b is None:
            b2b = self.a2b[self.b2a]

            revmask = (b2b != self.b2revb.unsqueeze(1).repeat(1, b2b.size(1))).long()
            self.b2b = b2b * revmask

        return self.b2b

    def get_a2a(self) -> torch.LongTensor:
        """
        Computes (if necessary) and returns a mapping from each atom index to all neighboring atom indices.

        :return: A PyTorch tensor containing the mapping from each bond index to all the incodming bond indices.
        """
        if self.a2a is None:





            a2neia=[]
            for incoming_bondIdList in self.a2b:
                neia=[]
                for incoming_bondId in incoming_bondIdList:
                    neia.append(self.b2a[incoming_bondId])
                a2neia.append(neia)
            self.a2a=a2neia

        return self.a2a

def mol2graph(smiles_batch: List[str],
              args: Namespace) -> BatchMolGraph:
    """
    Converts a list of SMILES strings to a BatchMolGraph containing the batch of molecular graphs.

    :param smiles_batch: A list of SMILES strings.
    :param args: Arguments.
    :return: A BatchMolGraph containing the combined molecular graph for the molecules
    """
    mol_graphs = []
    for smiles in smiles_batch:
        if smiles in SMILES_TO_GRAPH:
            mol_graph = SMILES_TO_GRAPH[smiles]
        else:
            mol_graph = MolGraph(smiles, args)
            if not args.no_cache:
                SMILES_TO_GRAPH[smiles] = mol_graph
        mol_graphs.append(mol_graph)

    return BatchMolGraph(mol_graphs, args)

class MPNEncoder(nn.Module):
    """A message passing neural network for encoding a molecule."""

    def __init__(self, args: Namespace, atom_fdim: int, bond_fdim: int):
        """Initializes the MPNEncoder.

        :param args: Arguments.
        :param atom_fdim: Atom features dimension.
        :param bond_fdim: Bond features dimension.
        """
        super(MPNEncoder, self).__init__()
        self.atom_fdim = atom_fdim
        self.bond_fdim = bond_fdim
        self.hidden_size = args.hidden_size
        self.bias = args.bias
        self.depth = args.depth
        self.dropout = args.dropout
        self.layers_per_message = args.layers_per_message
        self.undirected = args.undirected
        self.atom_messages = args.atom_messages
        self.features_only = args.features_only
        self.use_input_features = args.use_input_features
        self.normalize_messages=args.normalize_messages
        self.args = args
        self.diff_depth_weights=args.diff_depth_weights
        self.layer_norm=args.layer_norm

        self.attention = args.attention
        self.message_attention = args.message_attention
        self.global_attention = args.global_attention
        self.message_attention_heads = args.message_attention_heads
        self.sumstyle=args.sumstyle
        if self.features_only:
            return

        if args.layer_norm:
            self.layer_norm = nn.LayerNorm(self.hidden_size)


        self.dropout_layer = nn.Dropout(p=self.dropout)


        self.act_func = get_activation_function(args.activation)


        self.cached_zero_vector = nn.Parameter(torch.zeros(self.hidden_size), requires_grad=False)


        input_dim = self.atom_fdim if self.atom_messages else self.bond_fdim

        self.W_i = nn.Linear(input_dim, self.hidden_size, bias=self.bias)

        if self.message_attention:
            self.num_heads = self.message_attention_heads
            w_h_input_size = self.num_heads * self.hidden_size
            self.W_ma = nn.ModuleList([nn.Linear(self.hidden_size, self.hidden_size, bias=self.bias)
                                       for _ in range(self.num_heads)])
        if self.global_attention:
            self.W_ga1 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
            self.W_ga2 = nn.Linear(self.hidden_size, self.hidden_size)

        if self.atom_messages:
            w_h_input_size = self.hidden_size + self.bond_fdim
        else:
            w_h_input_size = self.hidden_size


        if args.diff_depth_weights:
            print(f'per depth with {self.layers_per_message} liner layers to per message')
            modulList=[nn.Linear(w_h_input_size, self.hidden_size, bias=self.bias)]
            modulList.extend(
                nn.ModuleList([nn.Linear(self.hidden_size, self.hidden_size, bias=self.bias) for _ in range(self.depth - 1)])
            )
            self.W_h=nn.Sequential(*modulList)
        else:

            print(f'Shared weight matrix across depths')
            self.W_h = nn.Linear(w_h_input_size, self.hidden_size, bias=self.bias)
        if self.sumstyle==True:
            self.W_ah= nn.Linear(self.atom_fdim, self.hidden_size)
            self.W_o = nn.Linear(self.hidden_size, self.hidden_size)
        else:
            self.W_o = nn.Linear(self.atom_fdim + self.hidden_size, self.hidden_size)

        if self.attention:
            self.W_a = nn.Linear(self.hidden_size, self.hidden_size, bias=self.bias)
            self.W_b = nn.Linear(self.hidden_size, self.hidden_size)


    def forward(self,
                mol_graph: BatchMolGraph,
                features_batch: List[np.ndarray] = None,
                viz_dir: str = None) -> torch.FloatTensor:
        """
        Encodes a batch of molecular graphs.

        :param mol_graph: A BatchMolGraph representing a batch of molecular graphs.
        :param features_batch: A list of ndarrays containing additional features.
        :return: A PyTorch tensor of shape (num_molecules, hidden_size) containing the encoding of each molecule.
        """
        if self.use_input_features:
            features_batch = torch.from_numpy(np.stack(features_batch)).float()

            if self.args.cuda:
                features_batch = features_batch.cuda()

            if self.features_only:
                return features_batch

        f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope = mol_graph.get_components()

        if self.atom_messages:
            a2a = mol_graph.get_a2a()

        if self.args.cuda or next(self.parameters()).is_cuda:
            f_atoms, f_bonds, a2b, b2a, b2revb = f_atoms.cuda(), f_bonds.cuda(), a2b.cuda(), b2a.cuda(), b2revb.cuda()

            if self.atom_messages:
                a2a = a2a.cuda()


        if self.atom_messages:
            input = self.W_i(f_atoms)
        else:
            input = self.W_i(f_bonds)

        message = self.act_func(input)

        if self.message_attention:
            b2b = mol_graph.get_b2b()
            if next(self.parameters()).is_cuda:
                b2b = b2b.cuda()
            message_attention_mask = (b2b != 0).float()
        if self.global_attention:
            global_attention_mask = torch.zeros(mol_graph.n_bonds, mol_graph.n_bonds)
            for start, length in b_scope:
                for i in range(start, start + length):
                    global_attention_mask[i, start:start + length] = 1
            if next(self.parameters()).is_cuda:
                global_attention_mask = global_attention_mask.cuda()


        for depth in range(self.depth - 1):
            if self.undirected:
                message = (message + message[b2revb]) / 2

            if self.atom_messages:
                nei_a_message = index_select_ND(message, a2a)
                nei_f_bonds = index_select_ND(f_bonds, a2b)
                nei_message = torch.cat((nei_a_message, nei_f_bonds), dim=2)
                message = nei_message.sum(dim=1)
            if self.message_attention:

                nei_message = index_select_ND(message, b2b)
                message = message.unsqueeze(1).repeat((1, nei_message.size(1), 1))
                attention_scores = [(self.W_ma[i](nei_message) * message).sum(dim=2)
                                    for i in range(self.num_heads)]
                attention_scores = [attention_scores[i] * message_attention_mask + (1 - message_attention_mask) * (-1e+20)
                                    for i in range(self.num_heads)]
                attention_weights = [F.softmax(attention_scores[i], dim=1)
                                     for i in range(self.num_heads)]
                message_components = [nei_message * attention_weights[i].unsqueeze(2).repeat((1, 1, self.hidden_size))
                                      for i in range(self.num_heads)]
                message_components = [component.sum(dim=1) for component in message_components]
                message = torch.cat(message_components, dim=1)
            else:


                nei_a_message = index_select_ND(message, a2b)
                a_message = nei_a_message.sum(dim=1)
                rev_message = message[b2revb]
                message = a_message[b2a] - rev_message


            message = self.W_h(message)
            message = self.act_func(input + message)

            if self.normalize_messages:
                message = message / message.norm(dim=1, keepdim=True)
            if self.global_attention:
                attention_scores = torch.matmul(self.W_ga1(message), message.t())
                attention_scores = attention_scores * global_attention_mask + (1 - global_attention_mask) * (-1e+20)
                attention_weights = F.softmax(attention_scores, dim=1)
                attention_hiddens = torch.matmul(attention_weights, message)
                attention_hiddens = self.act_func(self.W_ga2(attention_hiddens))
                attention_hiddens = self.dropout_layer(attention_hiddens)

                message = attention_hiddens
                if viz_dir is not None:
                    visualize_bond_attention(viz_dir, mol_graph, attention_weights, depth)
            if self.layer_norm:
                message = self.layer_norm(message)

            message = self.dropout_layer(message)


        a2x = a2a if self.atom_messages else a2b

        nei_a_message = index_select_ND(message, a2x)
        a_message = nei_a_message.sum(dim=1)

        if self.sumstyle==True:
            a_input =self.W_ah(f_atoms) + a_message
        else:
            a_input = torch.cat([f_atoms, a_message], dim=1)
        atom_hiddens = self.act_func(self.W_o(a_input))
        atom_hiddens = self.dropout_layer(atom_hiddens)


        mol_vecs = []
        for i, (a_start, a_size) in enumerate(a_scope):
            if a_size == 0:
                mol_vecs.append(self.cached_zero_vector)
            else:
                cur_hiddens = atom_hiddens.narrow(0, a_start, a_size)
                if self.attention:
                        att_w = torch.matmul(self.W_a(cur_hiddens), cur_hiddens.t())
                        att_w = F.softmax(att_w, dim=1)
                        att_hiddens = torch.matmul(att_w, cur_hiddens)
                        att_hiddens = self.act_func(self.W_b(att_hiddens))
                        att_hiddens = self.dropout_layer(att_hiddens)
                        mol_vec = (cur_hiddens + att_hiddens)
                        if viz_dir is not None:
                            visualize_atom_attention(viz_dir, mol_graph.smiles_batch[i], a_size, att_w)
                else:
                    mol_vec = cur_hiddens
                mol_vec = mol_vec.sum(dim=0) / a_size
                mol_vecs.append(mol_vec)

        mol_vecs = torch.stack(mol_vecs, dim=0)

        if self.use_input_features:
            features_batch = features_batch.to(mol_vecs)
            if len(features_batch.shape) == 1:
                features_batch = features_batch.view([1,features_batch.shape[0]])
            mol_vecs = torch.cat([mol_vecs, features_batch], dim=1)

        return mol_vecs

class MPN(nn.Module):
    """A message passing neural network for encoding a molecule."""

    def __init__(self,
                 args: Namespace,
                 graph_input: bool = False):
        """
        Initializes the MPN.

        :param args: Arguments.
        :param atom_fdim: Atom features dimension.
        :param bond_fdim: Bond features dimension.
        :param graph_input: If true, expects BatchMolGraph as input. Otherwise expects a list of smiles strings as input.
        """
        super(MPN, self).__init__()
        self.args = args
        self.atom_fdim = ATOM_FDIM
        self.bond_fdim = BOND_FDIM +(not args.atom_messages) *  self.atom_fdim

        self.graph_input = graph_input
        self.encoder = MPNEncoder(self.args, self.atom_fdim, self.bond_fdim)

    def forward(self,
                batch: Union[List[str], BatchMolGraph],
                features_batch: List[np.ndarray] = None) -> torch.FloatTensor:
        """
        Encodes a batch of molecular SMILES strings.

        :param batch: A list of SMILES strings or a BatchMolGraph (if self.graph_input is True).
        :param features_batch: A list of ndarrays containing additional features.
        :return: A PyTorch tensor of shape (num_molecules, hidden_size) containing the encoding of each molecule.
        """
        if not self.graph_input:
            batch = mol2graph(batch, self.args)

        output = self.encoder.forward(batch, features_batch)

        return output

    def viz_attention(self, batch: Union[List[str], BatchMolGraph],
                      features_batch: List[np.ndarray] = None,
                      viz_dir: str = None):
        """
        Visualizes attention weights for a batch of molecular SMILES strings
        :param viz_dir: Directory in which to save visualized attention weights.
        :param batch: A list of SMILES strings or a BatchMolGraph (if self.graph_input).
        :param features_batch: A list of ndarrays containing additional features.
        """
        if not self.graph_input:
            batch = mol2graph(batch, self.args)

        self.encoder.forward(batch, features_batch, viz_dir=viz_dir)
        print(f'usei++++++++++++++++++++++++++++++++++++++++viz_attention')
