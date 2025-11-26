import io
import pathlib
import warnings
import numpy as np
import torch as th
from gymnasium import spaces
import torch.nn.functional as F
from typing import Any, ClassVar, Dict, List, Optional, Type, TypeVar, Union
import time

from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env.patch_gym import _convert_space
from stable_baselines3.common.save_util import load_from_zip_file, recursive_setattr
from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3.common.utils import (
    get_system_info,
    update_learning_rate,
    explained_variance,
    update_learning_rate,
    obs_as_tensor,
)

from sb3_contrib import MaskablePPO
from sb3_contrib.ppo_mask.policies import CnnPolicy, MlpPolicy, MultiInputPolicy
from sb3_contrib.common.maskable.utils import get_action_masks, is_masking_supported
from sb3_contrib.common.maskable.buffers import (
    MaskableDictRolloutBuffer,
    MaskableRolloutBuffer,
)

# from model.custom_policy import CustomMlp, CustomGNN
import matplotlib.pyplot as plt

SelfBaseAlgorithm = TypeVar("SelfBaseAlgorithm", bound="BaseAlgorithm")
SelfMaskablePPO = TypeVar("SelfMaskablePPO", bound="MaskablePPO")


class CustomMaskablePPO(MaskablePPO):

    policy_aliases: ClassVar[Dict[str, Type[BasePolicy]]] = {
        "MlpPolicy": MlpPolicy,
        "CnnPolicy": CnnPolicy,
        "MultiInputPolicy": MultiInputPolicy,
    }

    def __init__(
        self,
        # focus:bool = True,
        # focus_coef:float = 1.0,
        *args,
        **kwargs,
    ):
        # self.focus = focus
        # self.focus_coef = focus_coef
        super().__init__(*args, **kwargs)
        # Profiling variables
        self.total_rollout_time = 0.0
        self.total_train_time = 0.0
        self.rollout_count = 0
        self.train_count = 0

    def _update_learning_rate(
        self, optimizers: Union[List[th.optim.Optimizer], th.optim.Optimizer]
    ) -> None:
        """
        Update the optimizers learning rate using the current learning rate schedule
        and the current progress remaining (from 1 to 0).

        :param optimizers:
            An optimizer or a list of optimizers.
        """
        # Log the current learning rate
        self.logger.record(
            "train/learning_rate", self.lr_schedule(self._current_progress_remaining)
        )

        if not isinstance(optimizers, list):
            optimizers = [optimizers]
        for optimizer in optimizers:
            update_learning_rate(
                optimizer, self.lr_schedule(self._current_progress_remaining)
            )

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Profiling: record start time
        train_start_time = time.time()

        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        # focus_losses = []
        # NContrast_losses = []

        continue_training = True

        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                # focus_loss = None
                # Ncontrast_loss = None
                # if isinstance(self.policy.features_extractor, CustomMlp):
                #     values, log_prob, entropy, Ncontrast_loss = self.policy.customMLP_evaluate_actions(
                #         rollout_data.observations,
                #         actions,
                #         action_masks=rollout_data.action_masks,
                #         adj=self.policy.features_extractor.edge_index if self.ncontrast_coef > 0 else None,
                #     )
                # if self.focus:
                #     values, log_prob, entropy, focus_loss = self.policy.customGNN_evaluate_actions(
                #         rollout_data.observations,
                #         actions,
                #         action_masks=rollout_data.action_masks,
                #     )
                # else:
                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations,
                    actions,
                    action_masks=rollout_data.action_masks,
                )

                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                if self.normalize_advantage:
                    advantages = (advantages - advantages.mean()) / (
                        advantages.std() + 1e-8
                    )

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(
                    ratio, 1 - clip_range, 1 + clip_range
                )
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the different between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)
                entropy_losses.append(entropy_loss.item())

                # Focus component loss mse sigmoid(score)<=>(mask->sum_node_wise)
                # if focus_loss is None:
                #     focus_loss = th.zeros_like(entropy_loss, device=entropy_loss.device)
                # else:
                #     focus_losses.append(focus_loss.item())

                # Ncontrast component
                # if Ncontrast_loss is None:
                #     Ncontrast_loss = th.zeros_like(entropy_loss, device=entropy_loss.device)
                # else:
                #     NContrast_losses.append(Ncontrast_loss.item())

                loss = (
                    policy_loss
                    + self.ent_coef * entropy_loss
                    + self.vf_coef * value_loss
                )  # + focus_loss * self.focus_coef

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = (
                        th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    )
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(
                            f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}"
                        )
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()

                def save_grad(name: str, param):
                    def draw_gradient(params, file_path: str = None):
                        def tolist(tensor: th.Tensor, name):
                            flattened_list = []
                            v = tensor.view(-1)
                            if not th.all(th.isnan(v) == False):
                                print(f"{name}: nan in tensor")
                            for elem in v:
                                flattened_list.append(elem.item())
                            return flattened_list

                        # grads = [param.grad.item() for param in params] # to scalar
                        grads = []
                        name = file_path.split("/")[-1].split(".")[0]
                        for p in params:
                            if p.grad is not None:
                                grads.extend(tolist(p.grad, name))  # to scalar
                        min_ = min(grads) if len(grads) > 0 else 0.0
                        max_ = max(grads) if len(grads) > 0 else 0.0
                        # print("{} gradient range [{},{}] len:{}".format(name, min_, max_, len(grads)))
                        plt.plot(range(len(grads)), grads)
                        plt.xlabel("Parameter Index")
                        plt.ylabel("Gradient Value")
                        plt.title("Gradient Range[{:.3e}-{:.3e}]".format(min_, max_))
                        plt.savefig(file_path)
                        plt.close()

                    base_path = f"out/img/"
                    draw_gradient(param, base_path + name + ".png")

                # save_grad('features_extractor', self.policy.features_extractor.parameters())
                # save_grad('mlp_extractor', self.policy.mlp_extractor.parameters())
                # save_grad('action_net', self.policy.action_net.parameters())
                # save_grad('value_net', self.policy.value_net.parameters())
                # print('draw over!')

                # Clip grad norm
                th.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm
                )
                self.policy.optimizer.step()

            if not continue_training:
                break

        self._n_updates += self.n_epochs
        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten()
        )

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)

        # if len(focus_losses) > 0:
        #     self.logger.record("train/focus_loss", np.mean(focus_losses))

        # if len(NContrast_losses) > 0:
        #     self.logger.record("train/NContrast_loss", np.mean(NContrast_losses))

        # Profiling: record end time and update statistics
        train_end_time = time.time()
        train_elapsed = train_end_time - train_start_time
        self.total_train_time += train_elapsed
        self.train_count += 1

        # Log profiling information
        self.logger.record("profiling/train_time", train_elapsed)
        self.logger.record(
            "profiling/avg_train_time", self.total_train_time / self.train_count
        )
        self.logger.record("profiling/total_train_time", self.total_train_time)

        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

    # @classmethod
    # def load(  # noqa: C901
    #     cls: Type[SelfBaseAlgorithm],
    #     path: Union[str, pathlib.Path, io.BufferedIOBase],
    #     env: Optional[GymEnv] = None,
    #     device: Union[th.device, str] = "auto",
    #     custom_objects: Optional[Dict[str, Any]] = None,
    #     print_system_info: bool = False,
    #     force_reset: bool = True,
    #     **kwargs,
    # ) -> SelfBaseAlgorithm:
    #     """
    #     Load the model from a zip-file.
    #     Warning: ``load`` re-creates the model from scratch, it does not update it in-place!
    #     For an in-place load use ``set_parameters`` instead.

    #     :param path: path to the file (or a file-like) where to
    #         load the agent from
    #     :param env: the new environment to run the loaded model on
    #         (can be None if you only need prediction from a trained model) has priority over any saved environment
    #     :param device: Device on which the code should run.
    #     :param custom_objects: Dictionary of objects to replace
    #         upon loading. If a variable is present in this dictionary as a
    #         key, it will not be deserialized and the corresponding item
    #         will be used instead. Similar to custom_objects in
    #         ``keras.models.load_model``. Useful when you have an object in
    #         file that can not be deserialized.
    #     :param print_system_info: Whether to print system info from the saved model
    #         and the current system info (useful to debug loading issues)
    #     :param force_reset: Force call to ``reset()`` before training
    #         to avoid unexpected behavior.
    #         See https://github.com/DLR-RM/stable-baselines3/issues/597
    #     :param kwargs: extra arguments to change the model when loading
    #     :return: new model instance with loaded parameters
    #     """
    #     if print_system_info:
    #         print("== CURRENT SYSTEM INFO ==")
    #         get_system_info()

    #     data, params, pytorch_variables = load_from_zip_file(
    #         path,
    #         device=device,
    #         custom_objects=custom_objects,
    #         print_system_info=print_system_info,
    #     )

    #     assert data is not None, "No data found in the saved file"
    #     assert params is not None, "No params found in the saved file"

    #     # Remove stored device information and replace with ours
    #     if "policy_kwargs" in data:
    #         if "device" in data["policy_kwargs"]:
    #             del data["policy_kwargs"]["device"]
    #         # backward compatibility, convert to new format
    #         if (
    #             "net_arch" in data["policy_kwargs"]
    #             and len(data["policy_kwargs"]["net_arch"]) > 0
    #         ):
    #             saved_net_arch = data["policy_kwargs"]["net_arch"]
    #             if isinstance(saved_net_arch, list) and isinstance(
    #                 saved_net_arch[0], dict
    #             ):
    #                 data["policy_kwargs"]["net_arch"] = saved_net_arch[0]

    #     if (
    #         "policy_kwargs" in kwargs
    #         and kwargs["policy_kwargs"] != data["policy_kwargs"]
    #     ):
    #         raise ValueError(
    #             f"The specified policy kwargs do not equal the stored policy kwargs."
    #             f"Stored kwargs: {data['policy_kwargs']}, specified kwargs: {kwargs['policy_kwargs']}"
    #         )

    #     if "observation_space" not in data or "action_space" not in data:
    #         raise KeyError(
    #             "The observation_space and action_space were not given, can't verify new environments"
    #         )

    #     # Gym -> Gymnasium space conversion
    #     for key in {"observation_space", "action_space"}:
    #         data[key] = _convert_space(data[key])

    #     if env is not None:
    #         # Wrap first if needed
    #         env = cls._wrap_env(env, data["verbose"])
    #         # Check if given env is valid
    #         # check_for_correct_spaces(env, data["observation_space"], data["action_space"])
    #         # Discard `_last_obs`, this will force the env to reset before training
    #         # See issue https://github.com/DLR-RM/stable-baselines3/issues/597
    #         if force_reset and data is not None:
    #             data["_last_obs"] = None
    #         # `n_envs` must be updated. See issue https://github.com/DLR-RM/stable-baselines3/issues/1018
    #         if data is not None:
    #             data["n_envs"] = env.num_envs
    #     else:
    #         # Use stored env, if one exists. If not, continue as is (can be used for predict)
    #         if "env" in data:
    #             env = data["env"]

    #     model = cls(
    #         policy=data["policy_class"],
    #         env=env,
    #         device=device,
    #         _init_setup_model=False,  # type: ignore[call-arg]
    #     )
    #     # load parameters
    #     model.__dict__.update(data)
    #     model.__dict__.update(kwargs)
    #     model._setup_model()

    #     if isinstance(model.policy.features_extractor, CustomGNN):
    #         if model.policy.features_extractor.sp_tensor:
    #             model.policy.features_extractor.update_sp_edge_index()  # update sp_tensor and csr
    #         else:
    #             model.policy.features_extractor.update_edge_index2adj()  # update adj matrix
    #     try:
    #         # put state_dicts back in place
    #         model.set_parameters(params, exact_match=True, device=device)
    #     except RuntimeError as e:
    #         # Patch to load Policy saved using SB3 < 1.7.0
    #         # the error is probably due to old policy being loaded
    #         # See https://github.com/DLR-RM/stable-baselines3/issues/1233
    #         if "pi_features_extractor" in str(
    #             e
    #         ) and "Missing key(s) in state_dict" in str(e):
    #             model.set_parameters(params, exact_match=False, device=device)
    #             warnings.warn(
    #                 "You are probably loading a model saved with SB3 < 1.7.0, "
    #                 "we deactivated exact_match so you can save the model "
    #                 "again to avoid issues in the future "
    #                 "(see https://github.com/DLR-RM/stable-baselines3/issues/1233 for more info). "
    #                 f"Original error: {e} \n"
    #                 "Note: the model should still work fine, this only a warning."
    #             )
    #         else:
    #             raise e
    #     # put other pytorch variables back in place
    #     if pytorch_variables is not None:
    #         for name in pytorch_variables:
    #             # Skip if PyTorch variable was not defined (to ensure backward compatibility).
    #             # This happens when using SAC/TQC.
    #             # SAC has an entropy coefficient which can be fixed or optimized.
    #             # If it is optimized, an additional PyTorch variable `log_ent_coef` is defined,
    #             # otherwise it is initialized to `None`.
    #             if pytorch_variables[name] is None:
    #                 continue
    #             # Set the data attribute directly to avoid issue when using optimizers
    #             # See https://github.com/DLR-RM/stable-baselines3/issues/391
    #             recursive_setattr(model, f"{name}.data", pytorch_variables[name].data)

    #     # Sample gSDE exploration matrix, so it uses the right device
    #     # see issue #44
    #     if model.use_sde:
    #         model.policy.reset_noise()  # type: ignore[operator]
    #     return model

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
        use_masking: bool = True,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        This method is largely identical to the implementation found in the parent class.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_steps: Number of experiences to collect per environment
        :param use_masking: Whether or not to use invalid action masks during training
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        # Profiling: record start time
        rollout_start_time = time.time()

        assert isinstance(
            rollout_buffer, (MaskableRolloutBuffer, MaskableDictRolloutBuffer)
        ), "RolloutBuffer doesn't support action masking"
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)
        n_steps = 0
        action_masks = None
        rollout_buffer.reset()

        if use_masking and not is_masking_supported(env):
            raise ValueError(
                "Environment does not support action masking. Consider using ActionMasker wrapper"
            )

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                # rllib
                obs_tensor = obs_as_tensor(self._last_obs, self.device)

                # This is the only change related to invalid action masking
                if use_masking:
                    action_masks = get_action_masks(env)

                actions, values, log_probs = self.policy(
                    obs_tensor, action_masks=action_masks
                )

            actions = actions.cpu().numpy()
            new_obs, rewards, dones, infos = env.step(actions)

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstraping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(
                        infos[idx]["terminal_observation"]
                    )[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]
                    rewards[idx] += self.gamma * terminal_value

            rollout_buffer.add(
                self._last_obs,
                actions,
                rewards,
                self._last_episode_starts,
                values,
                log_probs,
                action_masks=action_masks,
            )
            self._last_obs = new_obs
            self._last_episode_starts = dones

        with th.no_grad():
            # Compute value for the last timestep
            # Masking is not needed here, the choice of action doesn't matter.
            # We only want the value of the current observation.
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.on_rollout_end()

        # Profiling: record end time and update statistics
        rollout_end_time = time.time()
        rollout_elapsed = rollout_end_time - rollout_start_time
        self.total_rollout_time += rollout_elapsed
        self.rollout_count += 1

        # Log profiling information
        self.logger.record("profiling/rollout_time", rollout_elapsed)
        self.logger.record(
            "profiling/avg_rollout_time", self.total_rollout_time / self.rollout_count
        )
        self.logger.record("profiling/total_rollout_time", self.total_rollout_time)

        # Log ratio statistics
        if self.total_train_time > 0:
            total_time = self.total_rollout_time + self.total_train_time
            self.logger.record(
                "profiling/rollout_ratio", self.total_rollout_time / total_time
            )
            self.logger.record(
                "profiling/train_ratio", self.total_train_time / total_time
            )
            self.logger.record("profiling/total_time", total_time)

        return True
