import mpn
from mpn import  MPN
from argparse import Namespace
import torch.nn as nn
from nn_utils import get_activation_function, initialize_weights
from nn_utils import create_var, cuda, move_dgl_to_cuda

def build_model(args: Namespace) -> nn.Module:
    """
    Builds a MoleculeModel, which is a message passing neural network + feed-forward layers.

    :param args: Arguments.
    :return: A MoleculeModel containing the MPN encoder along with final linear layers with parameters initialized.
    """
    output_size = args.num_tasks
    print(f'output_size is {args.num_tasks}')
    model = QSARmodel(args,classification= args.dataset_type == 'classification')
    model.create_encoder(args)
    model.create_ffn(args)
    initialize_weights(model)
    print(f'has initialize_weights')
    return model


class QSARmodel(nn.Module):
    def __init__(self, args ,classification: bool):
        super(QSARmodel, self).__init__()
        self.classification = classification
        print(f'in QSARmodel classification={classification} ')
        self.args=args
        if self.classification:
            self.sigmoid = nn.Sigmoid()

        self.hidden_size = args.hidden_size
        self.depth = args.depth



    def create_encoder(self, args):

        """
        Creates the message passing encoder for the model.
        """
        self.encoder = MPN(args)

    def create_ffn(self, args: Namespace):
        """
        Creates the feed-forward network for the model.
        :param args: Arguments.
        """
        if args.features_only:
            first_linear_dim = args.features_size
        else:
            first_linear_dim = args.hidden_size
            if args.use_input_features:
                first_linear_dim += args.features_dim
        dropout = nn.Dropout(args.dropout)
        activation = get_activation_function(args.activation)

        if args.ffn_num_layers == 1:
            ffn = [
                dropout,
                nn.Linear(first_linear_dim, args.num_tasks)
            ]
        else:
            ffn = [
                dropout,
                nn.Linear(first_linear_dim, args.ffn_hidden_size)
            ]
            for _ in range(args.ffn_num_layers - 2):
                ffn.extend([
                    activation,
                    dropout,
                    nn.Linear(args.ffn_hidden_size, args.ffn_hidden_size),
                ])
            ffn.extend([
                activation,
                dropout,
                nn.Linear(args.ffn_hidden_size, args.num_tasks),
            ])

        if args.dataset_type == 'classification':
            ffn.append(self.sigmoid)

        self.ffn = nn.Sequential(*ffn)

    def forward(self, mol_batch, beta=0):

        output = self.ffn(self.encoder(mol_batch))


        return output