"""Various sampling methods."""
from scipy import integrate
import torch

from .predictors import Predictor, PredictorRegistry, ReverseDiffusionPredictor
from .correctors import Corrector, CorrectorRegistry

import matplotlib.pyplot as plt
import numpy as np
import torch.fft


__all__ = [
    'PredictorRegistry', 'CorrectorRegistry', 'Predictor', 'Corrector',
    'get_sampler'
]


def to_flattened_numpy(x):
    """Flatten a torch tensor `x` and convert it to numpy."""
    return x.detach().cpu().numpy().reshape((-1,))


def from_flattened_numpy(x, shape):
    """Form a torch tensor with the given `shape` from a flattened numpy array `x`."""
    return torch.from_numpy(x.reshape(shape))


def get_pc_sampler(
    predictor_name, corrector_name, sde, score_fn, y,
    denoise=True, eps=4e-2, snr=0.1, corrector_steps=0, probability_flow: bool = False,
    **kwargs
):
    """Create a Predictor-Corrector (PC) sampler.

    Args:
        predictor_name: The name of a registered `sampling.Predictor`.
        corrector_name: The name of a registered `sampling.Corrector`.
        sde: An `sdes.SDE` object representing the forward SDE.
        score_fn: A function (typically learned model) that predicts the score.
        y: A `torch.Tensor`, representing the (non-white-)noisy starting point(s) to condition the prior on.
        denoise: If `True`, add one-step denoising to the final samples.
        eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.
        snr: The SNR to use for the corrector. 0.1 by default, and ignored for `NoneCorrector`.
        N: The number of reverse sampling steps. If `None`, uses the SDE's `N` property by default.

    Returns:
        A sampling function that returns samples and the number of function evaluations during sampling.
    """
    predictor_cls = PredictorRegistry.get_by_name(predictor_name)
    corrector_cls = CorrectorRegistry.get_by_name(corrector_name)
    predictor = predictor_cls(sde, score_fn, probability_flow=probability_flow,**kwargs)
    corrector = corrector_cls(sde, score_fn, snr=snr, n_steps=corrector_steps, **kwargs)

    def pc_sampler():
        """The PC sampler function."""
        with torch.no_grad():
            timesteps = torch.linspace(sde.T, eps, sde.N, device=y.device)   
            xt = sde.prior_sampling(y.shape, y).to(y.device)
            for i in range(sde.N):
                    t = timesteps[i]
                    if i != len(timesteps) - 1:
                        stepsize = t - timesteps[i+1]
                    else:
                        stepsize = timesteps[-1] 
                    vec_t = torch.ones(y.shape[0], device=y.device) * t
                    xt, xt_mean = corrector.update_fn(xt, vec_t, y)
                    xt, xt_mean = predictor.update_fn(xt, vec_t, y, stepsize)
            x_result = xt_mean if denoise else xt 
            ns = sde.N * (corrector.n_steps + 1)
            return x_result, ns
    
    return pc_sampler
