import torch
from typing import List, Optional, Tuple, Union
import numpy as np
import math
import einops
from tqdm import tqdm
import torch.nn.functional as F


# Code based on that from the diffusers library from https://huggingface.co/docs/diffusers/api/schedulers/multistep_dpm_solver
# Some slight modifications to work with our code base.
# Provides access to high-order ODE solvers proposed by Lu et al.


def rescale_zero_terminal_snr(betas):
    """
    Rescales betas to have zero terminal SNR Based on https://arxiv.org/pdf/2305.08891.pdf (Algorithm 1)
    Args:
        betas (`torch.FloatTensor`):
            the betas that the scheduler is being initialized with.
    Returns:
        `torch.FloatTensor`: rescaled betas with zero terminal SNR
    """
    # Convert betas to alphas_bar_sqrt
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_bar_sqrt = alphas_cumprod.sqrt()

    # Store old values.
    alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()
    alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()

    # Shift so the last timestep is zero.
    alphas_bar_sqrt -= alphas_bar_sqrt_T

    # Scale so the first timestep is back to the old value.
    alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)

    # Convert alphas_bar_sqrt to betas
    alphas_bar = alphas_bar_sqrt**2  # Revert sqrt
    alphas = alphas_bar[1:] / alphas_bar[:-1]  # Revert cumprod
    alphas = torch.cat([alphas_bar[0:1], alphas])
    betas = 1 - alphas

    return betas

# Copied from diffusers.schedulers.scheduling_ddpm.betas_for_alpha_bar
def betas_for_alpha_bar(
    num_diffusion_timesteps,
    max_beta=0.999,
    alpha_transform_type="cosine",
):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function, which defines the cumulative product of
    (1-beta) over time from t = [0,1].

    Contains a function alpha_bar that takes an argument t and transforms it to the cumulative product of (1-beta) up
    to that part of the diffusion process.


    Args:
        num_diffusion_timesteps (`int`): the number of betas to produce.
        max_beta (`float`): the maximum beta to use; use values lower than 1 to
                     prevent singularities.
        alpha_transform_type (`str`, *optional*, default to `cosine`): the type of noise schedule for alpha_bar.
                     Choose from `cosine` or `exp`

    Returns:
        betas (`np.ndarray`): the betas used by the scheduler to step the model outputs
    """
    if alpha_transform_type == "cosine":

        def alpha_bar_fn(t):
            return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2

    elif alpha_transform_type == "exp":

        def alpha_bar_fn(t):
            return math.exp(t * -12.0)

    else:
        raise ValueError(f"Unsupported alpha_tranform_type: {alpha_transform_type}")

    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar_fn(t2) / alpha_bar_fn(t1), max_beta))
    return torch.tensor(betas, dtype=torch.float32)


class DPMSolverMultiStepScheduler():
    def __init__(
        self,
        n_train_steps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = 'linear',
        solver_order: int = 2,
        prediction_type: str = 'epsilon',
        thresholding: bool = False,
        dynamic_thresholding_ratio: float = 0.995,
        sample_max_value: float = 1.0,
        algorithm_type: str = 'dpmsolver++',
        solver_type: str = 'midpoint',
        lower_order_final: bool = True,
        use_karras_sigmas: bool = False,
        lambda_min_clipped: float = -float('inf'),
        timestep_spacing: str = 'linspace',
        steps_offset: int = 0,
    ):
        self.n_train_steps = n_train_steps
        self.solver_order = solver_order
        self.prediction_type = prediction_type
        self.thresholding = thresholding
        self.dynamic_thresholding_ratio = dynamic_thresholding_ratio
        self.sample_max_value = sample_max_value
        self.algorithm_type = algorithm_type
        self.solver_type = solver_type
        self.lower_order_final = lower_order_final
        self.use_karras_sigmas = use_karras_sigmas
        self.lambda_min_clipped = lambda_min_clipped
        self.timestep_spacing = timestep_spacing
        self.steps_offset = steps_offset

        if beta_schedule == 'linear':
            self.betas = torch.linspace(beta_start, beta_end, n_train_steps, dtype=torch.float32)
        elif beta_schedule == 'scaled_linear':
            self.betas = torch.linspace(beta_start**0.5, beta_end**0.5, n_train_steps, dtype=torch.float32)**2
        elif beta_schedule == 'squaredcos_cap_v2':
            self.betas = betas_for_alpha_bar(n_train_steps)
        else:
            raise NotImplementedError(f"{beta_schedule} does is not implemented for {self.__class__}")

        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        # VP noise schedule
        self.alpha_t = torch.sqrt(self.alphas_cumprod)
        self.sigma_t = torch.sqrt(1 - self.alphas_cumprod)
        self.lambda_t = torch.log(self.alpha_t) - torch.log(self.sigma_t)

        self.init_noise_sigma = 1.0

        self.n_inference_steps = None
        timesteps = np.linspace(0, n_train_steps - 1, n_train_steps, dtype=np.float32)[::-1].copy()
        self.timesteps = torch.from_numpy(timesteps)
        self.model_outputs = [None] * solver_order
        self.lower_order_nums = 0

    def set_timesteps(self, n_inference_steps: int = None, device: Union[str, torch.device] = None):
        """
        Sets the discrete timesteps used for the diffusion chain (to be run before inference).

        Args:
            num_inference_steps (`int`):
                The number of diffusion steps used when generating samples with a pre-trained model.
            device (`str` or `torch.device`, *optional*):
                The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        """
        # Clipping the minimum of all lambda(t) for numerical stability.
        # This is critical for cosine (squaredcos_cap_v2) noise schedule.
        clipped_idx = torch.searchsorted(torch.flip(self.lambda_t, [0]), self.lambda_min_clipped)
        last_timestep = ((self.n_train_steps - clipped_idx).numpy()).item()

        # "linspace", "leading", "trailing" corresponds to annotation of Table 2. of https://arxiv.org/abs/2305.08891
        if self.timestep_spacing == "linspace":
            timesteps = (
                np.linspace(0, last_timestep - 1, n_inference_steps + 1).round()[::-1][:-1].copy().astype(np.int64)
            )
        elif self.timestep_spacing == "leading":
            step_ratio = last_timestep // (n_inference_steps + 1)
            # creates integer timesteps by multiplying by ratio
            # casting to int to avoid issues when num_inference_step is power of 3
            timesteps = (np.arange(0, n_inference_steps + 1) * step_ratio).round()[::-1][:-1].copy().astype(np.int64)
            timesteps += self.steps_offset
        elif self.timestep_spacing == "trailing":
            step_ratio = self.n_train_steps / n_inference_steps
            # creates integer timesteps by multiplying by ratio
            # casting to int to avoid issues when num_inference_step is power of 3
            timesteps = np.arange(last_timestep, 0, -step_ratio).round().copy().astype(np.int64)
            timesteps -= 1
        else:
            raise ValueError(
                f"{self.timestep_spacing} is not supported. Please make sure to choose one of 'linspace', 'leading' or 'trailing'."
            )

        sigmas = np.array(((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5)
        
        if self.use_karras_sigmas:
            log_sigmas = np.log(sigmas)
            sigmas = self._convert_to_karras(in_sigmas=sigmas, num_inference_steps=n_inference_steps)
            timesteps = np.array([self._sigma_to_t(sigma, log_sigmas) for sigma in sigmas]).round()
            timesteps = np.flip(timesteps).copy().astype(np.int64)

        self.sigmas = torch.from_numpy(sigmas)

        # when num_inference_steps == num_train_timesteps, we can end up with
        # duplicates in timesteps.
        _, unique_indices = np.unique(timesteps, return_index=True)
        timesteps = timesteps[np.sort(unique_indices)]

        self.timesteps = torch.from_numpy(timesteps).to(device)

        self.n_inference_steps = len(timesteps)

        self.model_outputs = [
            None,
        ] * self.solver_order

        self.lower_order_nums = 0

    def reset_sampler(self):
        self.model_outputs = [
            None,
        ] * self.solver_order

        self.lower_order_nums = 0

    # Copied from diffusers.schedulers.scheduling_ddpm.DDPMScheduler._threshold_sample
    def _threshold_sample(self, sample: torch.FloatTensor) -> torch.FloatTensor:
        """
        "Dynamic thresholding: At each sampling step we set s to a certain percentile absolute pixel value in xt0 (the
        prediction of x_0 at timestep t), and if s > 1, then we threshold xt0 to the range [-s, s] and then divide by
        s. Dynamic thresholding pushes saturated pixels (those near -1 and 1) inwards, thereby actively preventing
        pixels from saturation at each step. We find that dynamic thresholding results in significantly better
        photorealism as well as better image-text alignment, especially when using very large guidance weights."

        https://arxiv.org/abs/2205.11487
        """
        dtype = sample.dtype
        batch_size, channels, height, width = sample.shape

        if dtype not in (torch.float32, torch.float64):
            sample = sample.float()  # upcast for quantile calculation, and clamp not implemented for cpu half

        # Flatten sample for doing quantile calculation along each image
        sample = sample.reshape(batch_size, channels * height * width)

        abs_sample = sample.abs()  # "a certain percentile absolute pixel value"

        s = torch.quantile(abs_sample, self.dynamic_thresholding_ratio, dim=1)
        s = torch.clamp(
            s, min=1, max=self.sample_max_value
        )  # When clamped to min=1, equivalent to standard clipping to [-1, 1]

        s = s.unsqueeze(1)  # (batch_size, 1) because clamp will broadcast along dim=0
        sample = torch.clamp(sample, -s, s) / s  # "we threshold xt0 to the range [-s, s] and then divide by s"

        sample = sample.reshape(batch_size, channels, height, width)
        sample = sample.to(dtype)

        return sample

    # Copied from diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler._sigma_to_t
    def _sigma_to_t(self, sigma, log_sigmas):
        # get log sigma
        log_sigma = np.log(sigma)

        # get distribution
        dists = log_sigma - log_sigmas[:, np.newaxis]

        # get sigmas range
        low_idx = np.cumsum((dists >= 0), axis=0).argmax(axis=0).clip(max=log_sigmas.shape[0] - 2)
        high_idx = low_idx + 1

        low = log_sigmas[low_idx]
        high = log_sigmas[high_idx]

        # interpolate sigmas
        w = (low - log_sigma) / (low - high)
        w = np.clip(w, 0, 1)

        # transform interpolation to time range
        t = (1 - w) * low_idx + w * high_idx
        t = t.reshape(sigma.shape)
        return t

    # Copied from diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler._convert_to_karras
    def _convert_to_karras(self, in_sigmas: torch.FloatTensor, num_inference_steps) -> torch.FloatTensor:
        """Constructs the noise schedule of Karras et al. (2022)."""

        sigma_min: float = in_sigmas[-1].item()
        sigma_max: float = in_sigmas[0].item()

        rho = 7.0  # 7.0 is the value used in the paper
        ramp = np.linspace(0, 1, num_inference_steps)
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        return sigmas

    def convert_model_output(self, model_output: torch.FloatTensor, timestep: int, sample: torch.FloatTensor) -> torch.FloatTensor:
        if self.algorithm_type in ['dpmsolver++', 'sde-dpmsolver++']:
            if self.prediction_type == 'epsilon':
                alpha_t, sigma_t = self.alpha_t[timestep], self.sigma_t[timestep]
                x0_pred = (sample - sigma_t * model_output) / alpha_t
            elif self.prediction_type == 'sample':
                x0_pred = model_output
            elif self.prediction_type == 'v_prediction':
                alpha_t, sigma_t = self.alpha_t[timestep], self.sigma_t[timestep]
                x0_pred = alpha_t * sample - sigma_t * model_output
            else:
                raise ValueError('prediction_type is invalid')

            if self.thresholding:
                x0_pred = self._threshold_sample(x0_pred)

            return x0_pred

        elif self.algorithm_type in ['dpmsolver', 'sde-dpmsolver']:
            if self.prediction_type == 'epsilon':
                epsilon = model_output
            elif self.prediction_type == 'sample':
                alpha_t, sigma_t = self.alpha_t[timestep], self.sigma_t[timestep]
                epsilon = (sample - alpha_t * model_output) / sigma_t
            elif self.prediction_type == 'v_prediction':
                alpha_t, sigma_t = self.alpha_t[timestep], self.sigma_t[timestep]
                epsilon = alpha_t * model_output + sigma_t * sample
            else:
                raise ValueError('prediction_type is invalid')

            if self.thresholding:
                alpha_t, sigma_t = self.alpha_t[timestep], self.sigma_t[timestep]
                x0_pred = (sample - sigma_t * epsilon) / alpha_t
                x0_pred = self._threshold_sample(x0_pred)
                epsilon = (sample - alpha_t * x0_pred) / sigma_t

            return epsilon

    def add_noise(
        self,
        original_samples: torch.FloatTensor,
        noise: torch.FloatTensor,
        timesteps: torch.IntTensor,
    ) -> torch.FloatTensor:
        # Make sure alphas_cumprod and timestep have same device and dtype as original_samples
        alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device, dtype=original_samples.dtype)
        timesteps = timesteps.to(original_samples.device)

        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise

        return noisy_samples

    def dpm_solver_first_order_update(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        prev_timestep: int,
        sample: torch.FloatTensor,
        noise: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        """
        One step for the first-order DPMSolver (equivalent to DDIM).

        Args:
            model_output (`torch.FloatTensor`):
                The direct output from the learned diffusion model.
            timestep (`int`):
                The current discrete timestep in the diffusion chain.
            prev_timestep (`int`):
                The previous discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                A current instance of a sample created by the diffusion process.

        Returns:
            `torch.FloatTensor`:
                The sample tensor at the previous timestep.
        """
        lambda_t, lambda_s = self.lambda_t[prev_timestep], self.lambda_t[timestep]
        alpha_t, alpha_s = self.alpha_t[prev_timestep], self.alpha_t[timestep]
        sigma_t, sigma_s = self.sigma_t[prev_timestep], self.sigma_t[timestep]
        h = lambda_t - lambda_s

        if self.algorithm_type == "dpmsolver++":
            x_t = (sigma_t / sigma_s) * sample - (alpha_t * (torch.exp(-h) - 1.0)) * model_output
        elif self.algorithm_type == "dpmsolver":
            x_t = (alpha_t / alpha_s) * sample - (sigma_t * (torch.exp(h) - 1.0)) * model_output
        elif self.algorithm_type == "sde-dpmsolver++":
            assert noise is not None
            x_t = (
                (sigma_t / sigma_s * torch.exp(-h)) * sample
                + (alpha_t * (1 - torch.exp(-2.0 * h))) * model_output
                + sigma_t * torch.sqrt(1.0 - torch.exp(-2 * h)) * noise
            )
        elif self.algorithm_type == "sde-dpmsolver":
            assert noise is not None
            x_t = (
                (alpha_t / alpha_s) * sample
                - 2.0 * (sigma_t * (torch.exp(h) - 1.0)) * model_output
                + sigma_t * torch.sqrt(torch.exp(2 * h) - 1.0) * noise
            )
        return x_t

    def multistep_dpm_solver_second_order_update(
        self,
        model_output_list: List[torch.FloatTensor],
        timestep_list: List[int],
        prev_timestep: int,
        sample: torch.FloatTensor,
        noise: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        """
        One step for the second-order multistep DPMSolver.

        Args:
            model_output_list (`List[torch.FloatTensor]`):
                The direct outputs from learned diffusion model at current and latter timesteps.
            timestep (`int`):
                The current and latter discrete timestep in the diffusion chain.
            prev_timestep (`int`):
                The previous discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                A current instance of a sample created by the diffusion process.

        Returns:
            `torch.FloatTensor`:
                The sample tensor at the previous timestep.
        """
        t, s0, s1 = prev_timestep, timestep_list[-1], timestep_list[-2]
        m0, m1 = model_output_list[-1], model_output_list[-2]
        lambda_t, lambda_s0, lambda_s1 = self.lambda_t[t], self.lambda_t[s0], self.lambda_t[s1]
        alpha_t, alpha_s0 = self.alpha_t[t], self.alpha_t[s0]
        sigma_t, sigma_s0 = self.sigma_t[t], self.sigma_t[s0]
        h, h_0 = lambda_t - lambda_s0, lambda_s0 - lambda_s1
        r0 = h_0 / h
        D0, D1 = m0, (1.0 / r0) * (m0 - m1)
        if self.algorithm_type == "dpmsolver++":
            # See https://arxiv.org/abs/2211.01095 for detailed derivations
            if self.solver_type == "midpoint":
                x_t = (
                    (sigma_t / sigma_s0) * sample
                    - (alpha_t * (torch.exp(-h) - 1.0)) * D0
                    - 0.5 * (alpha_t * (torch.exp(-h) - 1.0)) * D1
                )
            elif self.solver_type == "heun":
                x_t = (
                    (sigma_t / sigma_s0) * sample
                    - (alpha_t * (torch.exp(-h) - 1.0)) * D0
                    + (alpha_t * ((torch.exp(-h) - 1.0) / h + 1.0)) * D1
                )
        elif self.algorithm_type == "dpmsolver":
            # See https://arxiv.org/abs/2206.00927 for detailed derivations
            if self.solver_type == "midpoint":
                x_t = (
                    (alpha_t / alpha_s0) * sample
                    - (sigma_t * (torch.exp(h) - 1.0)) * D0
                    - 0.5 * (sigma_t * (torch.exp(h) - 1.0)) * D1
                )
            elif self.solver_type == "heun":
                x_t = (
                    (alpha_t / alpha_s0) * sample
                    - (sigma_t * (torch.exp(h) - 1.0)) * D0
                    - (sigma_t * ((torch.exp(h) - 1.0) / h - 1.0)) * D1
                )
        elif self.algorithm_type == "sde-dpmsolver++":
            assert noise is not None
            if self.solver_type == "midpoint":
                x_t = (
                    (sigma_t / sigma_s0 * torch.exp(-h)) * sample
                    + (alpha_t * (1 - torch.exp(-2.0 * h))) * D0
                    + 0.5 * (alpha_t * (1 - torch.exp(-2.0 * h))) * D1
                    + sigma_t * torch.sqrt(1.0 - torch.exp(-2 * h)) * noise
                )
            elif self.solver_type == "heun":
                x_t = (
                    (sigma_t / sigma_s0 * torch.exp(-h)) * sample
                    + (alpha_t * (1 - torch.exp(-2.0 * h))) * D0
                    + (alpha_t * ((1.0 - torch.exp(-2.0 * h)) / (-2.0 * h) + 1.0)) * D1
                    + sigma_t * torch.sqrt(1.0 - torch.exp(-2 * h)) * noise
                )
        elif self.algorithm_type == "sde-dpmsolver":
            assert noise is not None
            if self.solver_type == "midpoint":
                x_t = (
                    (alpha_t / alpha_s0) * sample
                    - 2.0 * (sigma_t * (torch.exp(h) - 1.0)) * D0
                    - (sigma_t * (torch.exp(h) - 1.0)) * D1
                    + sigma_t * torch.sqrt(torch.exp(2 * h) - 1.0) * noise
                )
            elif self.solver_type == "heun":
                x_t = (
                    (alpha_t / alpha_s0) * sample
                    - 2.0 * (sigma_t * (torch.exp(h) - 1.0)) * D0
                    - 2.0 * (sigma_t * ((torch.exp(h) - 1.0) / h - 1.0)) * D1
                    + sigma_t * torch.sqrt(torch.exp(2 * h) - 1.0) * noise
                )
        return x_t

    def multistep_dpm_solver_third_order_update(
        self,
        model_output_list: List[torch.FloatTensor],
        timestep_list: List[int],
        prev_timestep: int,
        sample: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """
        One step for the third-order multistep DPMSolver.

        Args:
            model_output_list (`List[torch.FloatTensor]`):
                The direct outputs from learned diffusion model at current and latter timesteps.
            timestep (`int`):
                The current and latter discrete timestep in the diffusion chain.
            prev_timestep (`int`):
                The previous discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                A current instance of a sample created by diffusion process.

        Returns:
            `torch.FloatTensor`:
                The sample tensor at the previous timestep.
        """
        t, s0, s1, s2 = prev_timestep, timestep_list[-1], timestep_list[-2], timestep_list[-3]
        m0, m1, m2 = model_output_list[-1], model_output_list[-2], model_output_list[-3]
        lambda_t, lambda_s0, lambda_s1, lambda_s2 = (
            self.lambda_t[t],
            self.lambda_t[s0],
            self.lambda_t[s1],
            self.lambda_t[s2],
        )
        alpha_t, alpha_s0 = self.alpha_t[t], self.alpha_t[s0]
        sigma_t, sigma_s0 = self.sigma_t[t], self.sigma_t[s0]
        h, h_0, h_1 = lambda_t - lambda_s0, lambda_s0 - lambda_s1, lambda_s1 - lambda_s2
        r0, r1 = h_0 / h, h_1 / h
        D0 = m0
        D1_0, D1_1 = (1.0 / r0) * (m0 - m1), (1.0 / r1) * (m1 - m2)
        D1 = D1_0 + (r0 / (r0 + r1)) * (D1_0 - D1_1)
        D2 = (1.0 / (r0 + r1)) * (D1_0 - D1_1)
        if self.algorithm_type == "dpmsolver++":
            # See https://arxiv.org/abs/2206.00927 for detailed derivations
            x_t = (
                (sigma_t / sigma_s0) * sample
                - (alpha_t * (torch.exp(-h) - 1.0)) * D0
                + (alpha_t * ((torch.exp(-h) - 1.0) / h + 1.0)) * D1
                - (alpha_t * ((torch.exp(-h) - 1.0 + h) / h**2 - 0.5)) * D2
            )
        elif self.algorithm_type == "dpmsolver":
            # See https://arxiv.org/abs/2206.00927 for detailed derivations
            x_t = (
                (alpha_t / alpha_s0) * sample
                - (sigma_t * (torch.exp(h) - 1.0)) * D0
                - (sigma_t * ((torch.exp(h) - 1.0) / h - 1.0)) * D1
                - (sigma_t * ((torch.exp(h) - 1.0 - h) / h**2 - 0.5)) * D2
            )
        return x_t

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        sample: torch.FloatTensor,
        generator=None,
        noise=None,
    ) -> torch.FloatTensor:
        if self.n_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )

        if isinstance(timestep, torch.Tensor):
            timestep = timestep.to(self.timesteps.device)
        step_index = (self.timesteps == timestep).nonzero()
        if len(step_index) == 0:
            step_index = len(self.timesteps) - 1
        else:
            step_index = step_index.item()
        prev_timestep = 0 if step_index == len(self.timesteps) - 1 else self.timesteps[step_index + 1]
        lower_order_final = (
            (step_index == len(self.timesteps) - 1) and self.lower_order_final and len(self.timesteps) < 15
        )
        lower_order_second = (
            (step_index == len(self.timesteps) - 2) and self.lower_order_final and len(self.timesteps) < 15
        )

        model_output = self.convert_model_output(model_output, timestep, sample)
        for i in range(self.solver_order - 1):
            self.model_outputs[i] = self.model_outputs[i + 1]
        self.model_outputs[-1] = model_output

        if self.algorithm_type in ["sde-dpmsolver", "sde-dpmsolver++"]:
            if noise is None:
                noise = torch.randn(model_output.shape, generator=generator, device=model_output.device, dtype=model_output.dtype)
        else:
            noise = None

        if self.solver_order == 1 or self.lower_order_nums < 1 or lower_order_final:
            prev_sample = self.dpm_solver_first_order_update(
                model_output, timestep, prev_timestep, sample, noise=noise
            )
        elif self.solver_order == 2 or self.lower_order_nums < 2 or lower_order_second:
            timestep_list = [self.timesteps[step_index - 1], timestep]
            prev_sample = self.multistep_dpm_solver_second_order_update(
                self.model_outputs, timestep_list, prev_timestep, sample, noise=noise
            )
        else:
            timestep_list = [self.timesteps[step_index - 2], self.timesteps[step_index - 1], timestep]
            prev_sample = self.multistep_dpm_solver_third_order_update(
                self.model_outputs, timestep_list, prev_timestep, sample
            )

        if self.lower_order_nums < self.solver_order:
            self.lower_order_nums += 1

        return prev_sample

    def reverse_step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        sample: torch.FloatTensor,
        generator=None,
        noise=None,
    ) -> torch.FloatTensor:
        if self.n_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )

        if isinstance(timestep, torch.Tensor):
            timestep = timestep.to(self.timesteps.device)
        step_index = (self.timesteps == timestep).nonzero()
        if len(step_index) == 0:
            step_index = 0
        else:
            step_index = step_index.item()

        next_timestep = self.timesteps[0] if step_index == 0 else self.timesteps[step_index - 1]

        lower_order_final = (
            (step_index == len(self.timesteps) - 1) and self.lower_order_final and len(self.timesteps) < 15
        )

        if self.algorithm_type in ["sde-dpmsolver", "sde-dpmsolver++"]:
            if noise is None:
                noise = torch.randn(model_output.shape, generator=generator, device=model_output.device, dtype=model_output.dtype)
        else:
            noise = None

        model_output = self.convert_model_output(model_output, timestep, sample)
        for i in range(self.solver_order - 1):
            self.model_outputs[i] = self.model_outputs[i + 1]
        self.model_outputs[-1] = model_output

        if self.solver_order == 1 or self.lower_order_nums < 1 or lower_order_final:
            prev_sample = self.dpm_solver_first_order_update(
                model_output, timestep, next_timestep, sample, noise=noise
            )
        else:
            timestep_list = [self.timesteps[step_index + 1], timestep]
            prev_sample = self.multistep_dpm_solver_second_order_update(
                self.model_outputs, timestep_list, next_timestep, sample, noise=noise
            )

        if self.lower_order_nums < self.solver_order:
            self.lower_order_nums += 1

        return prev_sample
