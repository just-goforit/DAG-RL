import numpy as np
import gymnasium as gym
from typing import Union, Optional
from env.env_const import *
from gymnasium import spaces
from env.graph import DAG_Graph, operator_info, naive_Conv2d_DAG


class RedBluePebbleGameEnv(gym.Env):
    def __init__(self, config: Optional[dict] = None):
        # 处理 config 为 None 的情况，提供默认空字典以符合 RLlib 接口规范
        if config is None:
            config = {}
        ## env name
        self.verbose: str = config.get("verbose", "")
        ## cache stat summary print flag
        self.summary = config.get("summary", False)
        ## observation output mode
        self.obs_mode = config.get("obs_mode", "vec")
        """
        box mode:
            node features will be aligned and zero-filled
        vec mode:
            node features will not be aligned, and will be flatten
        """
        if self.obs_mode not in ["vec", "box"]:
            raise ValueError(f"obs_mode:{self.obs_mode} should be in ['vec', 'box']")

        ## observation field
        # for `loc`` field only has node features, `glb` field add S
        self.obs_field = config.get("obs_field", "loc")
        if self.obs_field not in ["glb", "loc"]:
            raise ValueError(f"obs_mode:{self.obs_field} should be in ['glb', 'loc']")
        ## node actual action space size
        self.actual_node_action_size = ACTUAL_ACTION_SPACE_SIZE
        ## A max episode length: The episode will end after at most max_episode_len timesteps.
        ## Set to 0 or None for using no limit on the episode length.
        self.max_episode_len = config.get("max_episode_len", 0)
        ## Steps taken so far (after last reset).
        self.steps = 0
        ## reward dict
        self.reward_config = config.get("reward_config", ACTION_REWARD)
        ## Operator's DAG info
        op_info: operator_info = config.get("op_info", None)
        if op_info is None:
            raise ValueError(
                "op_info is required in config. Please provide operator_info "
                "via config['op_info']. Example: config={'op_info': operator_info(...)}"
            )
        self.operator_batch_size = op_info.batch_size
        self.operator_height = op_info.in_height
        self.operator_width = op_info.in_width
        self.operator_inplanes = op_info.inplanes
        self.operator_kernel_size = op_info.kernel_size
        self.operator_outplanes = op_info.outplanes
        self.operator_stride = op_info.stride
        self.operator_cache_limit = op_info.S
        ## static environment params
        # generate graph
        if op_info.env_id == "dConv2d":
            self.g = naive_Conv2d_DAG(op_info)
            # self.g is origin graph's backup for reset
        else:
            raise NotImplementedError()
        # mid node list
        self.mid_nodes_index = self.g.input_num + self.g.output_num
        self._edges = self.g.edge_index.copy().T  # edges src->dst
        self._edges = self._edges[self._edges[:, 0].argsort()]  # sort by src
        self._re_edges = self.g.edge_index.copy().T
        self._re_edges[:, [0, 1]] = self._re_edges[:, [1, 0]]  # reverse edges dst->src
        self._re_edges = self._re_edges[self._re_edges[:, 0].argsort()]  # sort by dst
        self._predecessors = []  # to cache predecessors
        self._successors = []  # to cache successors
        self._get_predecessors()
        self._get_successors()
        ## dynamic environment params
        self.rewd_sum = 0
        self.S = self.operator_cache_limit
        self.state = DAG_Graph(copy=self.g)  # raw features
        self.compute_counts = np.zeros(
            self.state.node_num, dtype=np.uint16
        )  # record node computed times
        self.fin_counts = 0
        self.mid_out = []  # mid nodes which have been output, decides game over or not
        self.valid_action_map = np.zeros(
            (self.state.node_num, self.actual_node_action_size), dtype=np.bool_
        )
        self._reset_valid_action_map()
        ## Action space.
        self.action_space_aligned = config.get(
            "align_action_space", False
        )  # algined action space size for all types of node
        # self.aligned_size = max(AGENT_IN_NODE_ACTION_SIZE, AGENT_OUT_NODE_ACTION_SIZE, AGENT_MID_NODE_ACTION_SIZE)
        # self.action_space = spaces.Discrete(self.g.input_num * AGENT_IN_NODE_ACTION_SIZE +
        #                                     self.g.output_num * AGENT_OUT_NODE_ACTION_SIZE +
        #                                     (self.g.node_num - self.g.input_num - self.g.output_num) * AGENT_MID_NODE_ACTION_SIZE) if not self.action_space_aligned else spaces.Discrete(self.g.node_num * self.aligned_size)

        # bacause [LOAD, STORE, DELETE, COMPUTE] is mutual exclusion with each other, so agent choose node is enough
        self.action_space = spaces.Discrete(self.g.node_num)

        ## Observation space from which to sample.

        if self.obs_mode == "vec":
            if self.obs_field == "glb":
                self.observation_space = spaces.Box(
                    low=0,
                    high=65536,
                    shape=(
                        ADD_FEATURE
                        + self.g.input_num * IN_NODE_FEATURES
                        + self.g.output_num * OUT_NODE_FEATURES
                        + self.g.mid_num * MID_NODE_FEATURES,
                    ),
                    dtype=np.float32,
                )
            else:
                self.observation_space = spaces.Box(
                    low=0,
                    high=1.0,
                    shape=(
                        self.g.input_num * IN_NODE_FEATURES
                        + self.g.output_num * OUT_NODE_FEATURES
                        + self.g.mid_num * MID_NODE_FEATURES,
                    ),
                    dtype=np.float32,
                )
        else:
            if self.obs_field == "glb":
                self.observation_space = spaces.Box(
                    low=0,
                    high=1.0,
                    shape=(
                        ADD_FEATURE
                        + self.g.node_num
                        * max(IN_NODE_FEATURES, MID_NODE_FEATURES, OUT_NODE_FEATURES),
                    ),
                    dtype=np.float32,
                )
            else:
                self.observation_space = spaces.Box(
                    low=0,
                    high=1.0,
                    shape=(
                        self.g.node_num
                        * max(IN_NODE_FEATURES, MID_NODE_FEATURES, OUT_NODE_FEATURES),
                    ),
                    dtype=np.float32,
                )
        ## constraint action
        # record node which is loaded but not be used to compute any successor
        self.load_lock_enable = config.get("load_lock_enable", False)
        self.load_lock = np.zeros(self.state.node_num, dtype=np.bool_)

        self.render_mode = config.get("render_mode", None)

        # autoly delete node_id
        self.del_lock_enable = config.get("del_lock_enable", False)
        self.del_lock = np.zeros(self.state.node_num, dtype=np.bool_)

        # early truncate
        self.action_truncate = config.get("action_truncate", False)

        ## cache data
        # to cache valid action mask give to agent at last step
        # avoid to calculate repeatedly
        self.cached_action_mask = None
        self.wraped_cached_action_mask = None

        ## output game progress to filename
        self.output_filename = config.get("filename", None)
        if self.verbose and self.output_filename is not None:
            with open(self.output_filename, "w") as file:
                file.truncate(0)

    def action_masks(self):
        """
        return current valid action space
        """
        mask = (
            self.wraped_cached_action_mask
            if self.wraped_cached_action_mask is not None
            else self._get_valid_action_mask()
        )
        return mask

    def _get_info(self, terminated=False, truncated=False, obs=None):
        info = {}
        # https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html#vecenv
        # vec env reset the env autoly, agent can only see last obs through info
        if terminated or truncated:
            info["terminal_observation"] = self._get_obs() if obs is None else obs
        info["TimeLimit.truncated"] = truncated and not terminated
        # Add action_mask for RLlib compatibility
        if self.cached_action_mask is not None:
            info["action_mask"] = self.action_masks()
        return info

    def _get_obs(self):
        """
        return current comprehensive state
        """
        # cat[cache_limit, in_node.flatten(), out_node.flatten(), mid_node.flatten()]
        # obs = {
        #     # 'S':float(self.S),
        #     'IN':self.state.x[:self.g.input_num, 0:IN_NODE_FEATURES].flatten(),
        #     'OUT':self.state.x[self.g.input_num:self.g.input_num+self.g.output_num, 0:OUT_NODE_FEATURES].flatten(),
        #     'MID':self.state.x[self.g.input_num+self.g.output_num:, 0:MID_NODE_FEATURES].flatten(),
        # }
        s = float(self.S) / self.operator_cache_limit
        if self.obs_mode == "vec":
            obs = np.concatenate(
                (
                    self.state.x[: self.g.input_num, 0:IN_NODE_FEATURES].flatten(),
                    self.state.x[
                        self.g.input_num : self.g.input_num + self.g.output_num,
                        0:OUT_NODE_FEATURES,
                    ].flatten(),
                    self.state.x[
                        self.g.input_num + self.g.output_num :, 0:MID_NODE_FEATURES
                    ].flatten(),
                )
            )
            if self.obs_field == "glb":
                obs = np.concatenate((np.array([s], dtype=np.float32), obs))
        else:
            # box mode
            obs = self.state.x.flatten()
            if self.obs_field == "glb":
                obs = np.concatenate((np.array([s], dtype=np.float32), obs))
        return obs

    def _human_action_decode(self, idx):
        """
        the way of reshape probas by human is [0, ACTION_SPACE_SIZE] --> [node_nums-1, ACTION_SPACE_SIZE]

        Return
        ------
        a tuple of (node_id, action)
        """
        return (idx // self.actual_node_action_size, idx % self.actual_node_action_size)

    def _agent_action_decode(self, idx):
        """
        the way of reshape probas by agent is
        [0, AGENT_IN_NODE_ACTION_SIZE) --> [input_nodes-1, AGENT_IN_NODE_ACTION_SIZE]

        [input_nodes, AGENT_OUT_NODE_ACTION_SIZE, ...] --> [input_nodes+output_nodes-1, AGENT_OUT_NODE_ACTION_SIZE]

        [input_nodes+output_nodes, AGENT_MID_NODE_ACTION_SIZE) --> [node_nums, AGENT_MID_NODE_ACTION_SIZE]
        Return
        ------
        a tuple of (node_id, action)
        """
        # if self.action_space_aligned:
        #     node_id = idx // self.aligned_size
        #     if node_id >= self.g.input_num + self.g.output_num:
        #         act = AGENT_MID_NODE_ACTS[idx % self.aligned_size] # STORE defaultly
        #         if act == STORE and self.state.x[node_id, NODE_MEMED] > 0:
        #             act = DELETE
        #         return (node_id, act)
        #     # assert idx % self.aligned_size <= 1 and "aligned action space is invalid should not be used"
        #     if node_id < self.g.input_num:
        #         return (node_id, AGENT_IN_NODE_ACTS[idx % self.aligned_size])
        #     return (node_id, AGENT_OUT_NODE_ACTS[idx % self.aligned_size])

        # if idx >= self.g.input_num * AGENT_IN_NODE_ACTION_SIZE:
        #     idx -= self.g.input_num * AGENT_IN_NODE_ACTION_SIZE
        #     if idx >= self.g.output_num * AGENT_OUT_NODE_ACTION_SIZE:
        #         idx -= self.g.output_num * AGENT_OUT_NODE_ACTION_SIZE
        #         node_id = self.mid_nodes_index + idx // AGENT_MID_NODE_ACTION_SIZE
        #         act = AGENT_MID_NODE_ACTS[idx % AGENT_MID_NODE_ACTION_SIZE] # STORE defaultly
        #         if act == STORE and self.state.x[node_id, NODE_MEMED] > 0:
        #             act = DELETE
        #         return (node_id, act)
        #     else:
        #         return (self.g.input_num + idx // AGENT_OUT_NODE_ACTION_SIZE,
        #                 AGENT_OUT_NODE_ACTS[idx % AGENT_OUT_NODE_ACTION_SIZE])
        # return (idx // AGENT_IN_NODE_ACTION_SIZE, AGENT_IN_NODE_ACTS[idx % AGENT_IN_NODE_ACTION_SIZE])

        if self.cached_action_mask is None:
            # RLlib/other frameworks may call `step()` without requesting action masks first,
            # ensure we always have the latest valid action mask cached.
            self._get_valid_action_mask()
        assert self.cached_action_mask is not None
        node_actions = self.cached_action_mask[idx]
        if idx >= self.g.input_num:
            if idx >= self.g.input_num + self.g.output_num:
                # mid node
                act = AGENT_MID_NODE_ACTS[
                    np.argmax(node_actions[AGENT_MID_NODE_ACTION_MASK])
                ]
                if node_actions[act] == False:
                    assert (
                        self.S == 0
                        and node_actions[DELETE] == True
                        and self.state.x[idx, NODE_MEMED] > 0
                    )
                    # has in mem trans to DELETE
                    act = DELETE
            else:
                act = AGENT_OUT_NODE_ACTS[
                    np.argmax(node_actions[AGENT_OUT_NODE_ACTION_MASK])
                ]
        else:
            act = AGENT_IN_NODE_ACTS[np.argmax(node_actions[AGENT_IN_NODE_ACTION_MASK])]

        if self.cached_action_mask[idx, act] == False:
            print("S:", self.S)
            print("idx:", idx)
            print("act", act)
            print("state:", self.state.x[idx])
            print(node_actions)

        assert node_actions[act] == True
        return idx, act

    def _action_decode(self, idx):
        if isinstance(idx, tuple):
            return idx
        if self.render_mode:
            return self._human_action_decode(idx)
        return self._agent_action_decode(idx)

    def _check_load_lock(self, node_id, act):
        # to make sure game will not truncate
        # when some nodes are locked(by [load lock])
        # will'not apply [del lock] on mask
        # but if next action is on [load lock], we should remove it from [load lock]
        if (act == STORE or act == DELETE) and (self.load_lock[node_id] == True):
            # self.load_lock[node_id] = False # remove node_id from [load lock], but may raise game stunk
            self.load_lock[:] = False  # just clear [load lock]

    def _check_del_lock(self, node_id):
        # to make sure game will not truncate
        # when some nodes are locked(by [del lock])
        # will'not apply [del lock] on mask
        # but if next action is on [del lock], we should remove it from [del lock]
        if self.del_lock[node_id] == True:
            # self.del_lock[node_id] = False # remove node_id from [del lock], but may raise game stunk
            # self.del_lock[:] = False # just clear [del lock]
            raise ValueError(
                "node_id:{} in del lock but be choosed, del lock should not stunk game".format(
                    node_id
                )
            )

    def _implicit_load(self, node_id):
        """
        auto load output node, do not raise steps add
        befor compute on output node(which has computed once), it should be loaded first autoly
        """
        # assert self._is_output(node_id) and "only output node can be loaded autoly"
        self.S -= 1
        self.state._change_node_color_by_action(
            node_id, LOAD
        )  # update node raw features
        self._update_valid_action_space(node_id, LOAD)
        # [load lock] is useless, because output node will be computed soon
        self._print_action(LOAD, node_id, "-")

    def _implicit_del(self, node_id):
        """
        autoly delete node_id, do not raise steps add
        """
        # assert self.state.x[node_id, NODE_CACHED] == 1 and "autoly delete node should be in cache"
        self.S += 1
        self.state._change_node_color_by_action(
            node_id, DELETE
        )  # update node raw features
        self._update_valid_action_space(node_id, DELETE)
        if self.del_lock_enable:
            self.del_lock[node_id] = True
        if self.verbose:
            self._print_action(DELETE, node_id, "-")

    def _auto_delete(self, node_id):
        """
        after agent do a COMPUTE on x, autoly delete some useless nodes in cache
        lock = []
        for ni in predecessors of x:
            if ni has no other successors except x:
                delete ni
            if ni has other successors except x:
                check if all of its successors has been computed once
                delete ni
            lock[ni] = True
        ...
        when get valid action mask:
            if there is any valid action:
                mask[lock[ni], :] = False
        """
        predecessors = self.get_predecessors(node_id)
        if self._is_midnode(node_id):
            # predecessors must be input node
            for predecessor in predecessors:
                successors = self.get_successors(predecessor)
                # successors mmust be mid node
                if len(successors) == 1 or np.all(
                    self.state.x[successors, NODE_COMPUTED] > 0
                ):
                    # predecessor has no other successors except node_id
                    # or
                    # predecessor has other successors except node_id and
                    # all of its successors has been computed once
                    self._implicit_del(predecessor)
        else:
            # assert self._is_output(node_id)
            # predecessors must be mid node
            for predecessor in predecessors:
                if self.state.x[predecessor, NODE_CACHED] > 0:
                    # assert self.state.x[predecessor, NODE_COMPUTED] > 0 and 'only never added dot item could add to output [del lock] must keep this'
                    # identify predecessor has contributed to output(inplace add to output)
                    self.state.x[predecessor, NODE_CONTRIBUTED] = 1
                    self._implicit_del(predecessor)

    def _auto_reduction(self, node_id, successor):
        """
        autoly reduct mid_node which is COMPUTED but not CONTRIBUTED, because it is in CACHE
        """
        # assert self._is_midnode(node_id) and "only mid node can be reduced autoly"
        # self.state._change_node_color_by_action(node_id, COMPUTE)
        # assert self.state.x[node_id, NODE_CACHED] < 1 and "node should not be in cache"
        # assert self.state.x[node_id, NODE_MEMED] > 0 and "node should be in memory"
        # assert self.state.x[node_id, NODE_COMPUTED] > 0 and "node should be computed"
        # assert self.state.x[node_id, NODE_CONTRIBUTED] < 1 and "node should not be contributed"
        self.state.x[node_id, NODE_CONTRIBUTED] = 1
        self._print_action(LOAD, node_id, "#")
        self._print_action(COMPUTE, successor, "#")
        self._print_action(DELETE, node_id, "#")
        if self.del_lock_enable:
            self.del_lock[node_id] = True

    def _auto_store(self, node_id):
        # auto store output
        # assert self._is_output(node_id)
        self.S += 1
        self.state._change_node_color_by_action(node_id, STORE)
        self._print_action(STORE, node_id, "-")
        # keep valid action map pure, update as it should be
        self.valid_action_map[node_id, DELETE] = self.valid_action_map[
            node_id, STORE
        ] = self.valid_action_map[node_id, LOAD] = False
        # self.valid_action_map[node_id, COMPUTE] = True # in _auto_delete will set it to False
        if self.del_lock_enable:
            self.del_lock[node_id] = True

    def _warp_action_mask(self, mask: np.ndarray):
        """
        convert action mask to agent format
        """
        # input_mask = mask[:self.g.input_num, AGENT_IN_NODE_ACTION_MASK]
        # if self.action_space_aligned:
        #     input_mask = np.concatenate((input_mask, np.zeros((self.g.input_num, 1), dtype=bool)), axis=1)
        # input_mask = input_mask.flatten()

        # output_mask = mask[self.g.input_num: self.g.input_num+self.g.output_num, :][:, AGENT_OUT_NODE_ACTION_MASK]
        # if self.action_space_aligned:
        #     output_mask = np.concatenate((output_mask, np.zeros((self.g.output_num, 1), dtype=bool)), axis=1)
        # output_mask = output_mask.flatten()
        # # Agent can't delete mid node which never be deleted because we assume that re-compute node cost more load/store than store-reload
        # # env will autoly delete useless mid node
        # # so agent can only store mid node(has in cache, and never be stored) / delete mid node(has in mem, and be loaded in)
        # mid_mask = mask[self.g.input_num+self.g.output_num:, :][:, AGENT_MID_NODE_ACTION_MASK]
        # mid_mask[:, AGENT_MID_NODE_ACTS_COL["STORE"]] = mid_mask[:,  AGENT_MID_NODE_ACTS_COL["STORE"]] | mask[self.g.input_num+self.g.output_num:, DELETE]
        # mid_mask = mid_mask.flatten()
        # return np.concatenate((input_mask, output_mask, mid_mask))

        input_mask = mask[: self.g.input_num, AGENT_IN_NODE_ACTION_MASK]
        input_mask = np.any(input_mask, axis=1)

        output_mask = mask[self.g.input_num : self.g.input_num + self.g.output_num, :][
            :, AGENT_OUT_NODE_ACTION_MASK
        ]
        output_mask = np.any(output_mask, axis=1)

        mid_mask = mask[self.g.input_num + self.g.output_num :, :][
            :, AGENT_MID_NODE_ACTION_MASK
        ]
        mid_mask[:, AGENT_MID_NODE_ACTS_COL["STORE"]] = (
            mid_mask[:, AGENT_MID_NODE_ACTS_COL["STORE"]]
            | mask[self.g.input_num + self.g.output_num :, DELETE]
        )
        mid_mask = np.any(mid_mask, axis=1)
        return np.concatenate((input_mask, output_mask, mid_mask))

    def _get_valid_action_mask(self):
        """
        ### attention
        delete after use
        """
        # assert self.S >= 0 # may useless
        tmp = self.valid_action_map.copy()

        if self.S == 0:
            # no S, no load/compute
            tmp[:, LOAD] = False
            tmp[np.where(self.state.x[:, NODE_CACHED] < 1)[0], COMPUTE] = (
                False  # when S == 0 may still compute on output(has in cache already)
            )
        else:
            # force to compute/load, order of delete/store will not affect optimal strategy
            tmp[:, DELETE] = tmp[:, STORE] = False

        # if self.S == 1:
        #     # for output, it may be computed at S==0 condition, when output node has already in cache, this space-cut technique is not contradicted with [del locl] and [load lock]
        #     # check whether has compute-able node which has in cache
        #     is_compute_able = np.any(
        #                         np.logical_and(
        #                             self.state.x[self.g.input_num: self.g.input_num+self.g.output_num, NODE_CACHED] > 0,
        #                             tmp[self.g.input_num: self.g.input_num+self.g.output_num, COMPUTE]))
        #     # if there aren't that nodes, then stop LOAD
        #     if not is_compute_able:
        #         # force to COMPUTE immediately
        #         tmp[:, LOAD] = False

        # lemma 2, when newly compute a mid-node and store it immediately, it should could be LOAD and COMPUTE, but this is not the optimal strategy, so we should avoid this situation
        # Encouragement oriented:
        tmp[
            np.concatenate(
                [
                    [False]
                    * (
                        self.g.input_num + self.g.output_num
                    ),  # do not affect input and output node
                    np.logical_and(
                        tmp[self.g.input_num + self.g.output_num :, LOAD],
                        tmp[self.g.input_num + self.g.output_num :, COMPUTE],
                    ),
                ]
            ),
            LOAD,
        ] = False
        # Criticism oriented:
        # tmp[np.concatenate([
        #         [False] * (self.g.input_num + self.g.output_num), # do not affect input and output node
        #         np.logical_and(tmp[self.g.input_num+self.g.output_num:, LOAD], tmp[self.g.input_num+self.g.output_num:, COMPUTE])
        #         ]), COMPUTE] = False

        if self.del_lock_enable:  # lemma 6
            tmp[self.del_lock, :] = False
            # if not np.any(tmp):
            #     # yqy
            #     print(self.del_lock)
            # assert np.any(tmp) and "[del lock] should nerver kill the game"

        if self.load_lock_enable:
            tmp_ = tmp.copy()
            tmp_[self.load_lock, DELETE] = tmp_[self.load_lock, STORE] = False
            if not np.any(tmp_) and not self.action_truncate:
                # if apply [load lock] on mask, and there is no valid action
                # do not apply [load lock] on
                tmp_ = tmp
            tmp = tmp_

        # force to compute firstly
        # if np.any(tmp[:, COMPUTE]):
        #     tmp[:, LOAD] = False

        self.cached_action_mask = tmp  # store a cached version instead of calculate at next step by _is_action_valid
        if self.render_mode:
            return self.cached_action_mask  # for human mode

        self.wraped_cached_action_mask = self._warp_action_mask(tmp)

        return self.wraped_cached_action_mask

    def _has_valid_action(self):
        """
        check whether there is any valid action
        attention: this function is also used to flash the cached action mask,
        for agent can get newly masks by cached data
        """
        return np.any(self._get_valid_action_mask())

    def _is_action_valid(self, action: Union[int, tuple]):
        """
        ### attention
        delete after use
        """
        # assert self.cached_action_mask is not None
        if self.render_mode:  # human mode
            if not isinstance(action, tuple):
                action = self._action_decode(action)
            if (
                action[0] >= self.state.node_num
                or action[1] >= self.actual_node_action_size
            ):
                return False
            return self.cached_action_mask[action[0], action[1]]
        # agent mode
        # assert not isinstance(action, tuple)
        if action >= self.action_space.n:
            return False
        return self.cached_action_mask[action]

    def _reset_valid_action_map(self):
        self.valid_action_map[:, :] = False
        self.valid_action_map[: self.g.input_num, LOAD] = True

    def reset(self, *, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        self.rewd_sum = 0
        self.steps = 0
        self.S = self.operator_cache_limit
        self.state = DAG_Graph(copy=self.g)  # reset state by copy
        self.compute_counts *= 0
        self.fin_counts = 0
        # self._reset_history()
        self._reset_valid_action_map()
        self.cached_action_mask = None
        self.wraped_cached_action_mask = None

        observation = self._get_obs()
        # 计算 action_mask 以便在 info 中包含它（RLlib 兼容性）
        self._get_valid_action_mask()
        info = self._get_info()

        if self.load_lock_enable:
            self.load_lock[:] = False

        if self.del_lock_enable:
            self.del_lock[:] = False

        return observation, info

    def _get_predecessors(self):
        """
        generate predecessors for each node, except for input node
        """
        self._predecessors = []
        for i in range(self.g.node_num - self.g.input_num):
            start_index = np.searchsorted(
                self._re_edges[:, 0], i + self.g.input_num, side="left"
            )
            end_index = np.searchsorted(
                self._re_edges[:, 0], i + self.g.input_num, side="right"
            )
            predecessors = self._re_edges[start_index:end_index, 1]
            self._predecessors.append(predecessors)

    def _get_successors(self):
        """
        generate successors for each node, except for output node
        """
        self._successors = []
        for i in range(self.g.node_num - self.g.output_num):
            node_id = i
            if i >= self.g.input_num:
                node_id += self.g.output_num  # jump over output node
            start_index = np.searchsorted(self._edges[:, 0], node_id, side="left")
            end_index = np.searchsorted(self._edges[:, 0], node_id, side="right")
            successors = self._edges[start_index:end_index, 1]
            self._successors.append(successors)

    def get_successors(self, node_id):
        if self._is_output(node_id):
            return np.array([], dtype=np.int64)
        if node_id < self.g.input_num:
            return self._successors[node_id]
        return self._successors[node_id - self.g.output_num]

    def get_predecessors(self, node_id):
        if self._is_input(node_id):
            return np.array([], dtype=np.int64)
        else:
            return self._predecessors[node_id - self.g.input_num]

    def _check_output_computability(self, node_id):
        """
        for output node
        when there is any predecessor in cache and never be computed, can it be computed again
        """
        if node_id < self.g.input_num:
            return False
        predecessors = self.get_predecessors(node_id)
        if np.any(
            (self.state.x[predecessors, NODE_COMPUTED] > 0)
            & (self.state.x[predecessors, NODE_CACHED] > 0)
        ):
            return True
        return False

    def _check_non_output_computability(self, node_id):
        """
        for mid node/ input node
        """
        # assert not self._is_output(node_id)
        if node_id < self.g.input_num or self.state._get_color(node_id) == RED:
            return False
        if self.state.x[node_id, NODE_COMPUTED] > 0:
            # mid node can only computed on itself once
            return False
        predecessors = self.get_predecessors(node_id)
        if np.all(self.state.x[predecessors, NODE_CACHED] > 0):
            # ver 2.0
            # dot node can only computed once and load later
            return True
        return False

    def _check_computability(self, node_id):
        """
        for mid node(dot opt)
        only when its predecessors have been in cache, can it be computed
        """
        if self._is_output(node_id):
            return self._check_output_computability(node_id)
        return self._check_non_output_computability(node_id)

    def _unlock_load_lock(self, node_id):
        """
        lemma 7: decrease vain load
        if node_id is computed, then its predecessors could be deleted/store and
        their load_lock should be set to False
        """
        if node_id < self.g.input_num:
            return
        # input node after load should alse used to computed at least one node before deleted or stored
        predecessors = self.get_predecessors(node_id)
        for predecessor in predecessors:
            self.load_lock[predecessor] = False

    def _is_output(self, node_id):
        return (
            node_id >= self.g.input_num
            and node_id < self.g.input_num + self.g.output_num
        )

    def _is_input(self, node_id):
        return node_id < self.g.input_num

    def _is_midnode(self, node_id):
        return node_id >= self.g.input_num + self.g.output_num

    def _print_action(self, act, node_id, prefix=""):
        # m = self.state.x
        # v = self._get_valid_action_mask()
        # for i in range(len(m)):
        #     print("{:>{}}:{}  {}".format(i, 3, m[i], v[i]))
        if self.verbose:
            if self.output_filename is None:
                print("{}Step:{} {}_{}".format(prefix, self.steps, ACTS[act], node_id))
            else:
                with open(self.output_filename, "a+") as file:
                    # '*' indicates that the action is autoly
                    file.write(
                        "{} {} {}\n".format(
                            ACTS[act], node_id, "+" if prefix == "" else "*"
                        )
                    )

    def _update_valid_action_space(self, node_id: int, act: int):
        # forbidden action:
        # put pebble on a node which has same color pebble(it may stuck)
        # red pebble numbers > S
        """
        scan the graph at each call is expensive,
        every action on the graph will influence itself and its successors (because of the DAG),
        so, scan the graph once at the beginning, and update the action space at each step.
        return a 2-dim matrix, shape: [nodes_num, 4(action_num)]
        +---------------+
        | T | F | T | T |
        | T | F | T | T |
        | T | F | T | T |
        | T | F | T | T |
        |      ...      |
        +---------------+
        """

        node_is_output = self._is_output(node_id)
        node_is_input = self._is_input(node_id) if not node_is_output else False
        node_is_mid = False if node_is_output or node_is_input else True
        # lemma/constraint
        # 1. if node has in mem, it should be deleted instead of store (V)
        # 2. if node is load-able and computea-able, it should be computed instead of load (LOAD-able==Load-able and not compute-able) (V), but this situation should not happen in optimal strategy, because STORE operation can be omit instead, for the dot is unique.
        # 3. if node in mem, it should not be deleted, given limitless memory (to cut search space) (V)
        # 4. if node has in cache, it should not be loaded/compute (to cut search space) (P)
        # => Env ver2.0: for output, it can be computed at some condition
        # 5. any node could not be operated continuously over 2 steps (X)
        # 6. auto delete useless node in cache after compute (V)
        # 7. a loaded node should nerver be deleted until its successor have been computed, except for input (V)
        # 8. a computed node should never be deleted until its successor have been computed, except for input (X)
        if act == LOAD:
            ## Load: blue->red|bule
            # for itself:
            #   1. delete-able [check]
            #   2. load-able   [false]
            #   3. store-able  [true]
            #   4. compute-able[false]
            # input and mid node handle load action as the same way
            # output node must store no delete
            self.valid_action_map[node_id, DELETE] = False if node_is_output else True
            self.valid_action_map[node_id, LOAD] = False
            self.valid_action_map[node_id, STORE] = (
                True if node_is_output else False
            )  # lemma 1, but env2.0: for output, it can be stored
            self.valid_action_map[node_id, COMPUTE] = (
                False
                if not node_is_output
                else self._check_output_computability(node_id)
            )  # lemma 4, but env2.0: for output, it may be computed-able
            # for it's successors:
            #   1. delete-able [still]
            #   2. load-able   [still]
            #   3. store-able  [still]
            #   4. compute-able[check] # avoid re-COMPUTE by check successor's color in _check_computability() for DAG is no-cycle and re-COMPUTE will get S-=2
            if not node_is_output:
                successors = self.get_successors(node_id)
                if node_is_input:
                    # successors are mid node
                    for successor in np.nditer(successors):
                        if self._check_non_output_computability(successor):
                            # predecessors' color == RED
                            self.valid_action_map[successor, COMPUTE] = True
                        # else:
                        #     assert self.valid_action_map[successor, COMPUTE] == False # useless
                else:
                    # assert node_is_mid
                    # assert self.state.x[node_id, NODE_CONTRIBUTED] == 0 and "del lock will force useless mid not be loaded"
                    for successor in np.nditer(successors):
                        self.valid_action_map[successor, COMPUTE] = True
                        # at least this node can be computed
            if self.load_lock_enable:
                self.load_lock[node_id] = True
        elif act == STORE:
            ## Store: red->blue
            # for itself:
            #   1. delete-able [check] * # 2023.12.4: input nodes can't be deleted from mem; 2023.12.10: to cut, donot allow delete node in mem
            #   2. load-able   [true]
            #   3. store-able  [false]
            #   4. compute-able[still] * # 2023.12.10: it may be re-COMPUTE by check predecessors
            self.valid_action_map[node_id, DELETE] = False  # lemma 3
            self.valid_action_map[node_id, LOAD] = True
            self.valid_action_map[node_id, STORE] = False
            # assert self.valid_action_map[node_id, COMPUTE] == self._check_computability(node_id) # this means STORE action never affect COMPUTE-able
            # self.valid_action_map[node_id, COMPUTE] will not change
            # for it's successors:
            #   1. delete-able [still]
            #   2. load-able   [still]
            #   3. store-able  [still]
            #   4. compute-able[false]
            if not node_is_output:
                successors = self.get_successors(node_id)
                # if node_is_input: # never happen
                #     # successors are mid node
                #     for successor in np.nditer(successors):
                #         self.valid_action_map[successor, COMPUTE] = False
                # else:
                # assert node_is_mid and len(successors) == 1
                # successors are out node
                for successor in np.nditer(successors):
                    self.valid_action_map[successor, COMPUTE] = (
                        self._check_output_computability(successor)
                    )
        elif act == COMPUTE:
            ## Compute: white->red, blue->red|blue
            # for itself:
            #   1. delete-able [true]
            #   2. load-able   [still] 2023.12.10: it may be re-LOAD by check predecessors, but to cut search space, we don't do this
            #   3. store-able  [true]
            #   4. compute-able[false]
            self.valid_action_map[node_id, DELETE] = (
                True if not node_is_output else False
            )  # ver 2.0
            self.valid_action_map[node_id, LOAD] = False
            self.valid_action_map[node_id, STORE] = (
                (self.state.x[node_id, NODE_MEMED] < 1) if not node_is_output else True
            )
            self.valid_action_map[node_id, COMPUTE] = (
                False  # ver 2.0 we do compute autoly for all accessible mid dot node
            )
            # for it's successors:
            #   1. delete-able [still]
            #   2. load-able   [still]
            #   3. store-able  [still]
            #   4. compute-able[check] *
            if node_is_mid:
                # only mid node can reach here
                # assert self.state.x[node_id, NODE_COMPUTED] == 1
                successors = self.get_successors(node_id)
                for successor in np.nditer(successors):
                    # sucessor is output node
                    self.valid_action_map[successor, COMPUTE] = True
                # else:
                #     self.valid_action_map[successor, COMPUTE] = False
            elif self.compute_counts[node_id] == self.g.output_indeg:
                self._auto_store(node_id)

            if self.load_lock_enable:
                # unlock node_id's predecessors, those can be deleted or stored
                self._unlock_load_lock(node_id)

            if self.del_lock_enable:
                if np.any(
                    self.state.x[
                        self.g.input_num : self.g.input_num + self.g.output_num,
                        NODE_COMPUTED,
                    ]
                    < 1
                ):
                    # forbid to delete autoly after finish
                    self._auto_delete(node_id)
        else:
            ## delete: red->white, red|blue->blue
            # for itself:
            #   1. delete-able [false]
            #   2. load-able   [check]
            #   3. store-able  [false]
            #   4. compute-able[check]
            self.valid_action_map[node_id, DELETE] = False  # lemma 3
            self.valid_action_map[node_id, LOAD] = self.state.x[node_id, NODE_MEMED] > 0
            self.valid_action_map[node_id, STORE] = False
            self.valid_action_map[node_id, COMPUTE] = (
                False  # for input node never be computed, for mid node, delete is autoly after sucessor(output) computed, before that, it's could only be stored
            )
            # for it's successors:
            #   1. delete-able [still]
            #   2. load-able   [still]
            #   3. store-able  [still]
            #   4. compute-able[false]
            # assert not node_is_output and 'output node should never be deleted'
            successors = self.get_successors(node_id)
            for successor in np.nditer(successors):
                # when node is input node, successors are mid node, so lack any of predecessors(node) compute-able turn to False
                # when node is mid node, successors are output node, delete are auto and all predecessors are used, so node[COMPUTE] will be False
                self.valid_action_map[successor, COMPUTE] = False

    def step(self, action: Union[int, tuple]):
        # if not self._is_action_valid(action):
        #     action_ = self._action_decode(action)
        #     raise ValueError(
        #         "Illegal action for steps:{} invalid:{}_{} code:{}".format(self.steps, ACTS[action_[1]], action_[0], action)
        #     )
        #     self.steps += 1
        #     return self._get_obs(), 0, False, False, {"error":"invalid action"} # just for unmasked env check

        action = self._action_decode(action)

        node_id, act = action

        # check whether need to remove node_id from [del lock]
        # del_lock will not stunk game
        if self.del_lock_enable and not self.action_truncate:
            self._check_del_lock(node_id)

        # check whether need to remove node_id from [load lock]
        if self.load_lock_enable and not self.action_truncate:
            self._check_load_lock(node_id, act)

        ## 0. step_n++
        self.steps += 1
        reward = 0  # current step's reward

        if self._is_output(node_id) and act == COMPUTE:
            # if COMPUTE on output node
            if (
                self.state.x[node_id, NODE_CACHED] < 1
                and self.state.x[node_id, NODE_MEMED] > 0
            ):
                # once computed, but stored in mem, then auto LOAD first
                # assert self.state.x[node_id, NODE_COMPUTED] < self.g.x[node_id, NODE_COMPUTED] and 'output node in mem and not in cache should have been computed'
                # perform auto LOAD
                self._implicit_load(node_id)
                reward += self.reward_config["LOAD"]

        ## 1. update cache size
        if act == LOAD:
            self.S -= 1
        elif act == COMPUTE:
            if self.state._get_color(node_id) != RED:
                # node id is not in cache
                # for mid node / first computed on output node
                self.S -= 1
        else:
            # STORE or DELETE
            self.S += 1

        ## 2. calculate reward
        if act == DELETE:
            reward += self.reward_config["DELETE"]
        elif act == LOAD:
            reward += self.reward_config["LOAD"]
        elif act == STORE:
            reward += self.reward_config["STORE"]
        else:  # COMPUTE
            if self._is_midnode(node_id):
                # assert self.state.x[node_id, NODE_COMPUTED] == 2 and "mid node can only compute itself once"
                self.state.x[node_id, NODE_COMPUTED] = 1
                reward += self.reward_config["COMPUTE"]
            else:
                # assert self._is_output(node_id)
                predecessors = self.get_predecessors(node_id)
                nums = np.count_nonzero(
                    (self.state.x[predecessors, NODE_COMPUTED] > 0)
                    & (self.state.x[predecessors, NODE_CACHED] > 0)
                    & (self.state.x[predecessors, NODE_CONTRIBUTED] < 1)
                )
                assert nums > 0
                # self.state.x[node_id, NODE_COMPUTED] -= nums
                self.compute_counts[node_id] += nums
                self.state.x[node_id, NODE_COMPUTED] = float(
                    self.compute_counts[node_id]
                ) / float(self.g.output_indeg)
                reward += self.reward_config["COMPUTE"] * nums

        reward += self.reward_config["TIME_COST"]

        ## 3. put/remove pebble
        self.state._change_node_color_by_action(
            node_id, act
        )  # update node raw features
        self._print_action(act=act, node_id=node_id)

        ## 4. update valid_action_space
        self._update_valid_action_space(node_id, act)

        ## 5. autoly load-compute-delete mid node(COMPUTED but not CACHED)
        if (
            act == COMPUTE
            and self._is_output(node_id)
            and self.compute_counts[node_id] < self.g.output_indeg
        ):
            in_mem_mask = (
                (self.state.x[predecessors, NODE_COMPUTED] > 0)
                & (self.state.x[predecessors, NODE_MEMED] > 0)
                & (self.state.x[predecessors, NODE_CONTRIBUTED] < 1)
            )
            in_mem_nodes = predecessors[in_mem_mask]
            for i in in_mem_nodes:
                self._auto_reduction(i, node_id)
                reward += self.reward_config["COMPUTE"]
                reward += self.reward_config["LOAD"]
                reward += self.reward_config["DELETE"]
                self.compute_counts[node_id] += 1
            if self.compute_counts[node_id] == self.g.output_indeg:
                # auto store output node
                self._auto_store(node_id)
        ## 6. terminal check
        terminated = False
        # if np.all(self.state.x[self.g.input_num:self.g.input_num+self.g.output_num, NODE_COMPUTED] > 1):
        if act == COMPUTE:
            if self.compute_counts[node_id] == self.g.output_indeg:
                self.fin_counts += 1
                reward += self.reward_config["STORE"]  # autoly store
                if self.fin_counts == self.g.output_num:
                    reward += self.reward_config["DONE"]
                    if self.summary:
                        print(
                            "Done reward sum:{:.3f} in {} steps".format(
                                self.rewd_sum + reward, self.steps
                            )
                        )
                    terminated = True
        ## 7. truncate check
        truncated = False
        if (
            not terminated
            and self.max_episode_len > 0
            and self.steps >= self.max_episode_len
        ):
            if self.summary:
                print(
                    "Truncated reward sum:{:.3f} at {} steps".format(
                        self.rewd_sum + reward, self.steps
                    )
                )
            truncated = True

        # if not truncated and not terminated:
        #     assert self._has_valid_action() == True

        self._get_valid_action_mask()  # must update action mask

        nxt_observation = self._get_obs()
        info = self._get_info(terminated, truncated, obs=nxt_observation)

        ## 8. cumulate reward
        self.rewd_sum += reward
        return nxt_observation, reward, terminated, truncated, info

    def render(self):
        """
        print the state of the environment
        """
        if self.render_mode and self.render_mode == "human":
            print("S:", self.S)
            m = self.state.x
            v = self._get_valid_action_mask()
            print("state: M S C   Del   Load  Store Compute")
            for i in range(len(m)):
                print("  {:>{}}:{} {}".format(i, 3, m[i], v[i]))
            print("")
        return
