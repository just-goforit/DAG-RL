# SHOW_ENV_FRQ = 0 # debug hand only
# node color use in environment
WHITE = 0
BLUE = 1
RED = 2
# node feature nums
IN_NODE_FEATURES = 2  # [NODE_MEMED, NODE_CACHED]
MID_NODE_FEATURES = 4 # [NODE_MEMED, NODE_CACHED, NODE_COMPUTED, NODE_CONTRIBUTED]
OUT_NODE_FEATURES = 3 # [NODE_MEMED, NODE_CACHED, NODE_COMPUTED]
ADD_FEATURE = 1 # include S
# For index node feature
NODE_MEMED=0
NODE_CACHED=1
NODE_COMPUTED=2
NODE_CONTRIBUTED=3 # only valid for mid node
# action space size
# ACTION_SPACE_SIZE = 4 # deprecated
ACTUAL_ACTION_SPACE_SIZE = 4 # only used for RedBluePebbleGameEnv maintain action mask
# ACTION SPACE FOR AGENT
'''
FOR  input node:

'''
AGENT_IN_NODE_ACTION_SIZE = 2 # LOAD DELETE # can never be deleted/compute
AGENT_MID_NODE_ACTION_SIZE = 3 # LOAD STORE COMPUTE # delete autoly
AGENT_OUT_NODE_ACTION_SIZE = 2 # LOAD STORE COMPUTE # can never be delete # ver 2.0
# action encode
DELETE = 0 
LOAD = 1
STORE = 2
COMPUTE = 3
# action decode
AGENT_IN_NODE_ACTION_MASK =[DELETE, LOAD]
AGENT_MID_NODE_ACTION_MASK=[LOAD, STORE, COMPUTE]
AGENT_OUT_NODE_ACTION_MASK=[STORE, COMPUTE]
AGENT_IN_NODE_ACTS={
    0:DELETE,
    1:LOAD}
AGENT_MID_NODE_ACTS={
    0:LOAD,
    1:STORE,
    2:COMPUTE}
AGENT_OUT_NODE_ACTS={
    0:STORE,
    1:COMPUTE # autoly load from mem if not in cache, except first time
}
AGENT_MID_NODE_ACTS_COL={
    "LOAD":0,
    "STORE":1,
    "COMPUTE":2}
ACTS={0:"DELETE",
      1:"LOAD",
      2:"STORE",
      3:"COMPUTE"}
"""
reward rules
"""
ACTION_REWARD = {
    "TIME_COST": 0,
    "DELETE": 0,
    "LOAD": -1,
    "STORE": -1,
    "COMPUTE": 0,
    "RECOMPUTE": 0,
    "REDUNDANT": 0, # penalty for redundant trap 2023.12.19, masked maybe better
    "DONE": 0       # reward for finish task
}
"""
Truncate
"""
MAX_EPISODE = 100
SHOW_ENV_FRQ = 1