from typing import Any, Dict, Optional, Type, TypeVar, Union
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule

# from stable_baselines3.common.policies import BasePolicy
# policy = BasePolicy()

# PyTorch

import numpy as np
import torch as th
from gym import spaces
from torch import nn
from torch.distributions import Categorical
import torch.nn.functional as F

from stable_baselines3.common.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    Distribution,
    MultiCategoricalDistribution,
    StateDependentNoiseDistribution,
    make_proba_distribution,
)

from stable_baselines3.common.preprocessing import get_action_dim, is_image_space, maybe_transpose, preprocess_obs
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    MlpExtractor,
    NatureCNN,
    create_mlp,
)

from stable_baselines3.common.utils import (
    check_for_correct_spaces,
    get_device,
    get_schedule_fn,
    get_system_info,
    set_random_seed,
    update_learning_rate,
)

from stable_baselines3.common.policies import BasePolicy, BaseModel, ActorCriticPolicy, ActorCriticCnnPolicy, MultiInputActorCriticPolicy

class ReinforcePolicy(ActorCriticPolicy):
    """We use the vanilla policy for actor critic algorithm. The only difference
    is that we won't be using the critic part - which could be scrapped...
    """
    pass

from collections import deque

SelfReinforce = TypeVar("SelfReinfroce", bound="Reinforce")


from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
class ReinforceAlgorithm(OnPolicyAlgorithm):
    """docstring for Reinforce"""

    policy_aliases: Dict[str, Type[BasePolicy]] = {
        "MlpPolicy": ActorCriticPolicy,
        "CnnPolicy": ActorCriticCnnPolicy,
        "MultiInputPolicy": MultiInputActorCriticPolicy,
    }

    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 7e-4,
        n_steps: int = 5,
        gamma: float = 0.99,
        gae_lambda: float = 1.0,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        rms_prop_eps: float = 1e-5,
        use_rms_prop: bool = True,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        normalize_advantage: bool = False,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):

        super().__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            seed=seed,
            _init_setup_model=False,
            supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
            ),
        )

        self.normalize_advantage = normalize_advantage

        # Update optimizer inside the policy if we want to use RMSProp
        # (original implementation) rather than Adam
        if use_rms_prop and "optimizer_class" not in self.policy_kwargs:
            self.policy_kwargs["optimizer_class"] = th.optim.RMSprop
            self.policy_kwargs["optimizer_kwargs"] = dict(alpha=0.99, eps=rms_prop_eps, weight_decay=0)

        if _init_setup_model:
            self._setup_model()

    def get_from_rolloutbuffer(self, x):
        x = self.rollout_buffer.swap_and_flatten(x)
        x = th.tensor(x, device=self.device)
        return x

    def train(self) -> None:
        """
        Update policy using the currently gathered
        rollout buffer (one gradient step over whole data).
        """

        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)

        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)

        # get log_prob with gradient
        actions = self.get_from_rolloutbuffer(self.rollout_buffer.actions)
        observations = self.get_from_rolloutbuffer(self.rollout_buffer.observations)
        if isinstance(self.action_space, spaces.Discrete):
            # Convert discrete action from float to long
            actions = actions.long().flatten()
        values, log_probs, entropy = self.policy.evaluate_actions(observations, actions)

        # local variables initialisation
        # Line 6 of pseudocode: calculate the return
        returns = deque(maxlen=self.rollout_buffer.buffer_size)
        policy_loss = []

        rewards = self.get_from_rolloutbuffer(self.rollout_buffer.rewards).cpu().numpy()
        episode_starts = self.get_from_rolloutbuffer(self.rollout_buffer.episode_starts).cpu().numpy()

        # loop over the whole buffer in reverse
        for t_step in reversed(range(self.rollout_buffer.buffer_size)):

            # Compute the discounted returns at each timestep, as
            # the sum of the gamma-discounted return at time t (G_t) + the reward at time t
            disc_return_t = (returns[0] if len(returns)>0 else 0)
            returns.appendleft(self.gamma*disc_return_t + rewards[t_step])

            # stop at the end of new episode
            if episode_starts[t_step] and not (t_step==self.rollout_buffer.buffer_size-1):

                ## standardization of the returns is employed to make training more stable
                eps = np.finfo(np.float32).eps.item()
                returns = th.tensor(np.array(returns))
                returns = (returns - returns.mean()) / (returns.std() + eps)

                # Line 7:
                for log_prob, disc_return in zip(log_probs[t_step:], returns):
                    policy_loss.append(-log_prob * disc_return.to(self.device))
                returns = deque(maxlen=self.rollout_buffer.buffer_size)

        # FIX: if policy loss not empty
        if policy_loss:
            policy_loss = th.cat(policy_loss).sum()

            # Line 8: PyTorch prefers gradient descent
            self.policy.optimizer.zero_grad()
            policy_loss.backward()
            self.policy.optimizer.step()


    def learn(
        self: SelfReinforce,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "Reinforce",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfReinforce:

        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )


if __name__ == '__main__':

    import gym
    import gym_pygame

    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3 import PPO, A2C, SAC, TD3, DQN

    env_id = "Pixelcopter-PLE-v0"
    env = gym.make(env_id)
    # env = make_vec_env(env_id, 1)
    from gym.wrappers import FrameStack
    env = FrameStack(env, 4)


    # ENV_ID = "Pixelcopter-PLE-v0"
    # DEFAULT_HYPERPARAMS = {
    #     "policy": "MlpPolicy",
    #     "env": env_id,
    # }

    # hyperparameters = {
    #     "n_steps": 1000,
    #     "gamma": 0.045,
    #     "learning_rate": 0.00085,
    #     "max_grad_norm": 0.645,
    #     "policy_kwargs": {
    #         "net_arch": {"pi": [64, 64], "vf": [64, 64]},
    #         "activation_fn": nn.ReLU,
    #     },
    # }

    # kwargs = DEFAULT_HYPERPARAMS.copy()
    # kwargs.update(hyperparameters)

    algo = ReinforceAlgorithm("MlpPolicy", env, n_steps=20, verbose=1).learn(total_timesteps=100000, log_interval=10)
    # algo = A2C("MlpPolicy", env, n_steps=100, verbose=1).learn(total_timesteps=100, log_interval=10)


    from stable_baselines3.common.evaluation import evaluate_policy
    eval_envs = gym.make(env_id)
    eval_envs = FrameStack(eval_envs, 4)

    mean_reward, std_reward = evaluate_policy(algo, eval_envs, n_eval_episodes=100, deterministic=True)
    print(f"Tuned Reinforce Mean episode reward: {mean_reward:.2f} +/- {std_reward:.2f}")
