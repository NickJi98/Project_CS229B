# Calculate temporal intensity

import numpy as np
import torch, sys

from my_func.read_model import nstpp_path
if nstpp_path not in sys.path:
    sys.path.append(nstpp_path)

from models.spatial.cnf import max_rms_norm
from models.spatial.attncnf import gaussian_loglik


def create_time_samples(event_times, t0, t1, npts_interval=5):
    
    # Number of events
    T = event_times.shape[0]
    
    # Time intervals
    # intervals = torch.diff(torch.cat((t0, event_times, t1)))
    intervals = torch.cat((t0, event_times, t1))
    intervals = intervals[1:] - intervals[:-1]
    
    # Create time samples
    base_samples = torch.linspace(0, 1, npts_interval+2)
    time_samples = intervals.view(-1, 1) @ base_samples.view(1, -1)
    time_samples = time_samples + torch.cat((t0, event_times)).view(-1, 1)
    
    return time_samples


def get_intensities(tpp_model, event_times, spatial_location, t0, t1, npts_interval=5):
    """
    Args:
        tpp_model: Temporal Point Process
        event_times: (T,)
        spatial_location: (T, D)
        t0: (1,)
        t1: (1,)
    """
    
    # Number of events
    T = event_times.shape[0]
    
    if not tpp_model.cond:
        # disable dependence on spatial sample.
        spatial_location = None
        
    init_state = tpp_model._init_state
    state = (
        torch.zeros(1).to(init_state),  # Lambda(t_0)
        init_state,
    )
    
    t0_i = t0 if torch.is_tensor(t0) else torch.tensor(t0)
    t0_i = t0_i.to(event_times)
    
    Lambda = []
    intensities = []
    prejump_hidden_states = []
    postjump_hidden_states = []
    
    for i in range(T):

        t1_i = event_times[i]
        state_traj = tpp_model.ode_solver.integrate(t0_i, t1_i, state, nlinspace=npts_interval+1, 
                                                    method="dopri5" if tpp_model.training else "dopri5")

        _Lambda, tpp_state = state_traj
        Lambda.append(_Lambda[-1])
        intensities.append(tpp_model.get_intensity(tpp_state).reshape(-1))

        cond = spatial_location[i] if spatial_location is not None else None
        postjump_tpp_state = tpp_model.hidden_state_dynamics.update_state(event_times[i], 
                                                                          tpp_state[-1], cond=cond)
        state = (_Lambda[-1], postjump_tpp_state)
        prejump_hidden_states.append(tpp_state[-1])
        postjump_hidden_states.append(postjump_tpp_state)

        # Track t0 as the last valid event time.
        t0_i = event_times[i]

    # Integrate from last time sample to t1.
    t1 = t1 if torch.is_tensor(t1) else torch.tensor(t1)
    t1 = t1.to(event_times)
    state_traj = tpp_model.ode_solver.integrate(t0_i, t1, state, nlinspace=npts_interval+1, 
                                                method="dopri5" if tpp_model.training else "dopri5")

    _, tpp_state = state_traj
    intensities.append(tpp_model.get_intensity(tpp_state).reshape(-1))
    
    intensities = torch.stack(intensities, dim=0) # (T+1, npts_interval+2)
    Lambda = torch.stack(Lambda, dim=0).reshape(-1)  # (T, )
    prejump_hidden_states = torch.stack(prejump_hidden_states, dim=0)  # (T, D)
    postjump_hidden_states = torch.stack(postjump_hidden_states, dim=0)  # (T, D)
    return intensities, Lambda, prejump_hidden_states, postjump_hidden_states


def read_lambda_npz(npz_files):

    all_times, all_locs, all_mags, all_pre_lambdas, all_post_lambdas = [], [], [], [], []

    for npz_file in npz_files:
        data = np.load(npz_file)
        all_times.append(data['all_times'])
        all_locs.append(data['all_locs'])
        all_mags.append(data['all_mags'])
        all_pre_lambdas.append(data['all_pre_lambdas'])
        all_post_lambdas.append(data['all_post_lambdas'])

    all_times = np.concatenate(all_times)
    all_locs = np.concatenate((all_locs))
    all_mags = np.concatenate(all_mags)
    all_pre_lambdas = np.concatenate(all_pre_lambdas)
    all_post_lambdas = np.concatenate(all_post_lambdas)
    
    return all_times, all_locs, all_mags, all_pre_lambdas, all_post_lambdas