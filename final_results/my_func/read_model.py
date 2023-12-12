# Read input model and training / validation / test log likelihood

import os, sys

# Adjust '..' based on your directory structure
nstpp_path = '/path/to/neural_stpp'
if nstpp_path not in sys.path:
    sys.path.append(nstpp_path)

import re, argparse
import numpy as np
import torch

from models import CombinedSpatiotemporalModel, JumpCNFSpatiotemporalModel, SelfAttentiveCNFSpatiotemporalModel, JumpGMMSpatiotemporalModel
from models.spatial import GaussianMixtureSpatialModel, IndependentCNF, JumpCNF, SelfAttentiveCNF
from models.spatial.cnf import TimeVariableCNF
from models.temporal import HomogeneousPoissonPointProcess, HawkesPointProcess, SelfCorrectingPointProcess, NeuralPointProcess
from models.temporal.neural import ACTFNS as TPP_ACTFNS
from models.temporal.neural import TimeVariableODE

# CPU or GPU setting
device = torch.device(f'cuda:{rank:d}' if torch.cuda.is_available() else 'cpu')


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def read_arguments(log_file):
    pattern = re.compile(r"Namespace\((.*?)\)")

    # Read the file line by line and extract the Namespace information
    with open(log_file, 'r') as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                namespace_str = match.group(1)
                namespace_args = re.findall(r"(\w+)\s*=\s*([^\s,]+)", namespace_str)
                args_dict = {}

                # Read input parameters
                for key, value in namespace_args:
                    value = value.strip('\'\"')
                    try:
                        value = int(value)
                    except ValueError:
                        try:
                            value = float(value)
                        except ValueError:
                            pass
                    args_dict[key] = value

                args = argparse.Namespace(**args_dict)
                break
    return args

def read_loglik(log_file):
    train_LL = AttrDict({
        "iter": [], "epoch": [],
        "lr": [],
        "temporal": np.empty((0, 2)),
        "spatial": np.empty((0, 2))
    })

    val_LL = AttrDict({
        "iter": [],
        "temporal": [],
        "spatial": []
    })

    test_LL = AttrDict({
        "iter": [],
        "temporal": [],
        "spatial": []
        })

    with open(log_file, 'r') as f:
        for line in f:
            if "Iter" in line and "[Test]" not in line:
                line = re.sub(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} - ", "", line)
                matches = re.findall(r"[-\d.]+", line)

                train_LL["iter"].append(int(matches[0]))
                train_LL["epoch"].append(int(matches[1]))
                train_LL["lr"].append(float(matches[2]))
                temporal_values = np.array([float(matches[4]), float(matches[5])])
                spatial_values = np.array([float(matches[6]), float(matches[7])])
                train_LL["temporal"] = np.vstack([train_LL["temporal"], temporal_values])
                train_LL["spatial"] = np.vstack([train_LL["spatial"], spatial_values])

            elif "[Test]" in line:
                line = re.sub(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} - ", "", line)
                matches = re.findall(r"[-\d.]+", line)

                val_LL["iter"].append(int(matches[0]))
                val_LL["temporal"].append(float(matches[1]))
                val_LL["spatial"].append(float(matches[2]))
                test_LL["iter"].append(int(matches[0]))
                test_LL["temporal"].append(float(matches[3]))
                test_LL["spatial"].append(float(matches[4]))

    val_LL.temporal, val_LL.spatial = np.array(val_LL.temporal), np.array(val_LL.spatial)
    test_LL.temporal, test_LL.spatial = np.array(test_LL.temporal), np.array(test_LL.spatial)

    return train_LL, val_LL, test_LL


def cast(tensor, device):
    return tensor.float().to(device)


def to_numpy(x):
    if torch.is_tensor(x):
        return x.cpu().detach().numpy()
    return [to_numpy(x_i) for x_i in x]


def get_dim(data):
    if data == "gmm":
        return 1
    elif data == "fmri" or data == 'eq_mag':
        return 3
    else:
        return 2


def get_t0_t1(data):
    if data == "citibike":
        return torch.tensor([0.0]), torch.tensor([24.0])
    elif data == "covid_nj_cases":
        return torch.tensor([0.0]), torch.tensor([7.0])
    elif data == "earthquakes_jp":
        return torch.tensor([0.0]), torch.tensor([30.0])
    elif data == "pinwheel":
        return torch.tensor([0.0]), torch.tensor([toy_datasets.END_TIME])
    elif data == "gmm":
        return torch.tensor([0.0]), torch.tensor([toy_datasets.END_TIME])
    elif data == "fmri":
        return torch.tensor([0.0]), torch.tensor([10.0])
    elif data == "eq_mag":
        return torch.tensor([0.0]), torch.tensor([30.0])
    else:
        raise ValueError(f"Unknown dataset {data}")


def create_model(args):
    x_dim = get_dim(args.data)

    if args.model == "jumpcnf" and args.tpp == "neural":
        model = JumpCNFSpatiotemporalModel(dim=x_dim, spatial_dim=args.spatial_dim,
                                           hidden_dims=list(map(int, args.hdims.split("-"))),
                                           tpp_hidden_dims=list(map(int, args.tpp_hdims.split("-"))),
                                           actfn=args.actfn,
                                           tpp_cond=args.tpp_cond,
                                           tpp_style=args.tpp_style,
                                           tpp_actfn=args.tpp_actfn,
                                           share_hidden=args.share_hidden,
                                           solve_reverse=args.solve_reverse,
                                           tol=args.tol,
                                           otreg_strength=args.otreg_strength,
                                           tpp_otreg_strength=args.tpp_otreg_strength,
                                           layer_type=args.layer_type,
                                           ).to(device)
    elif args.model == "attncnf" and args.tpp == "neural":
        model = SelfAttentiveCNFSpatiotemporalModel(dim=x_dim, spatial_dim=args.spatial_dim,
                                                    hidden_dims=list(map(int, args.hdims.split("-"))),
                                                    tpp_hidden_dims=list(map(int, args.tpp_hdims.split("-"))),
                                                    actfn=args.actfn,
                                                    tpp_cond=args.tpp_cond,
                                                    tpp_style=args.tpp_style,
                                                    tpp_actfn=args.tpp_actfn,
                                                    share_hidden=args.share_hidden,
                                                    solve_reverse=args.solve_reverse,
                                                    l2_attn=args.l2_attn,
                                                    tol=args.tol,
                                                    otreg_strength=args.otreg_strength,
                                                    tpp_otreg_strength=args.tpp_otreg_strength,
                                                    layer_type=args.layer_type,
                                                    lowvar_trace=not args.naive_hutch,
                                                    ).to(device)
    elif args.model == "cond_gmm" and args.tpp == "neural":
        model = JumpGMMSpatiotemporalModel(dim=x_dim, spatial_dim=args.spatial_dim,
                                           hidden_dims=list(map(int, args.hdims.split("-"))),
                                           tpp_hidden_dims=list(map(int, args.tpp_hdims.split("-"))),
                                           actfn=args.actfn,
                                           tpp_cond=args.tpp_cond,
                                           tpp_style=args.tpp_style,
                                           tpp_actfn=args.tpp_actfn,
                                           share_hidden=args.share_hidden,
                                           tol=args.tol,
                                           tpp_otreg_strength=args.tpp_otreg_strength,
                                           ).to(device)
    else:
        # Mix and match between spatial and temporal models.
        if args.tpp == "poisson":
            tpp_model = HomogeneousPoissonPointProcess()
        elif args.tpp == "hawkes":
            tpp_model = HawkesPointProcess()
        elif args.tpp == "correcting":
            tpp_model = SelfCorrectingPointProcess()
        elif args.tpp == "neural":
            tpp_hidden_dims = list(map(int, args.tpp_hdims.split("-")))
            tpp_model = NeuralPointProcess(
                cond_dim=x_dim, hidden_dims=tpp_hidden_dims, cond=args.tpp_cond, style=args.tpp_style, actfn=args.tpp_actfn,
                otreg_strength=args.tpp_otreg_strength, tol=args.tol)
        else:
            raise ValueError(f"Invalid tpp model {args.tpp}")

        if args.model == "gmm":
            model = CombinedSpatiotemporalModel(GaussianMixtureSpatialModel(), tpp_model).to(device)
        elif args.model == "cnf":
            model = CombinedSpatiotemporalModel(
                IndependentCNF(dim=x_dim, hidden_dims=list(map(int, args.hdims.split("-"))),
                               layer_type=args.layer_type, actfn=args.actfn, tol=args.tol, otreg_strength=args.otreg_strength,
                               squash_time=True),
                tpp_model).to(device)
        elif args.model == "tvcnf":
            model = CombinedSpatiotemporalModel(
                IndependentCNF(dim=x_dim, hidden_dims=list(map(int, args.hdims.split("-"))),
                               layer_type=args.layer_type, actfn=args.actfn, tol=args.tol, otreg_strength=args.otreg_strength),
                tpp_model).to(device)
        elif args.model == "jumpcnf":
            model = CombinedSpatiotemporalModel(
                JumpCNF(dim=x_dim, hidden_dims=list(map(int, args.hdims.split("-"))),
                        layer_type=args.layer_type, actfn=args.actfn, tol=args.tol, otreg_strength=args.otreg_strength),
                tpp_model).to(device)
        elif args.model == "attncnf":
            model = CombinedSpatiotemporalModel(
                SelfAttentiveCNF(dim=x_dim, hidden_dims=list(map(int, args.hdims.split("-"))),
                                 layer_type=args.layer_type, actfn=args.actfn, l2_attn=args.l2_attn, tol=args.tol, otreg_strength=args.otreg_strength),
                tpp_model).to(device)
        else:
            raise ValueError(f"Invalid model {args.model}")

    return model