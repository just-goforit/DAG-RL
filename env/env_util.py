import sys
sys.path.append('..')
from env.env_const import *
from colorama import Fore, Style
from env.RedBluePebbleGame import RedBluePebbleGameEnv

def play(config:dict, speeds:list=None, agent=None):
    env = RedBluePebbleGameEnv(config=config)
    done = False
    truncate = False
    reward_sum = 0
    i = 0
    if agent:
        agent.eval_()
    print(Fore.GREEN + '[HUMAN] Game start...' + Style.RESET_ALL)
    try:
        while not done and not truncate:
            # env.render()
            if speeds is not None and i < len(speeds):
                print(f'Step:{env.steps + 1} Node id: {speeds[i][0]} Action: {ACTS[speeds[i][1]]}', end=' ')
                node_id = speeds[i][0]
                act = speeds[i][1]
                i+=1
            else:
                env.render()
                # if agent:
                #     with torch.no_grad():
                #         probas, v = agent.predict(state2tensor(env.state.x), s2tensor(env.S), edge_index2tensor(env.state.edge_index))
                #     probas:np.ndarray = probas.cpu().data.numpy()
                #     probas_view = probas.reshape((-1, ACTION_SPACE_SIZE))
                #     valid_moves = env.get_valid_action_mask()
                #     probas_view[~valid_moves] = 0
                #     total = np.sum(probas)
                #     if total != 0:
                #         probas /= total
                #     sorted_indices = np.argsort(probas.flatten())[::-1]
                #     top_indices = sorted_indices[:k]
                #     print("Agent> v: {:.3f}".format(v.item()), end=' ')
                #     for ind in top_indices:
                #         n_id, act_ = env.probas_idx2action(ind)
                #         print('[{} {}]:{:.3f}'.format(ACTS[act_],n_id,probas[0, ind]), end='| ')
                print(f'Step:{env.steps + 1} Input your action |0:delete|1:load|2:store|3:compute|')
                node_id = int(input('Node id: '))
                act = int(input('Action: '))
            if env._is_action_valid((node_id, act)) == False:
                print(Fore.RED +'[HUMAN] Invalid action\n' + Style.RESET_ALL)
                continue
            _,  reward, done, truncate, info = env.step((node_id, act))
            reward_sum += reward
            print(f'Reward: {reward}')
            if info:
                print(f'Info: {info}')
    except KeyboardInterrupt:
        print("[HUMAN] interrupted")
    finally:
        print(f"[HUMAN] Reward sum: {reward_sum}")