import argparse
import gym
import os
import sys
import pickle
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from environment.custom_dubin import ContinuousDubinGym,DiscreteDubinGym
from utils import *
from models.mlp_policy import Policy
from models.mlp_critic import Value
from models.mlp_policy_disc import DiscretePolicy
from core.trpo import trpo_step
from core.common import estimate_advantages
from core.agent import Agent
import datetime
from torch.utils.tensorboard import SummaryWriter


parser = argparse.ArgumentParser(description='PyTorch TRPO example')
parser.add_argument('--model-path', metavar='G',
                    help='path of pre-trained model')
parser.add_argument('--render', action='store_true', default=False,
                    help='render the environment')
parser.add_argument('--log-std', type=float, default=-0.0, metavar='G',
                    help='log std for the policy (default: -0.0)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.95, metavar='G',
                    help='gae (default: 0.95)')
parser.add_argument('--l2-reg', type=float, default=1e-3, metavar='G',
                    help='l2 regularization regression (default: 1e-3)')
parser.add_argument('--max-kl', type=float, default=1e-2, metavar='G',
                    help='max kl value (default: 1e-2)')
parser.add_argument('--damping', type=float, default=1e-2, metavar='G',
                    help='damping (default: 1e-2)')
parser.add_argument('--num-threads', type=int, default=1, metavar='N',
                    help='number of threads for agent (default: 4)')
parser.add_argument('--seed', type=int, default=1, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--min-batch-size', type=int, default=2048, metavar='N',
                    help='minimal batch size per TRPO update (default: 2048)')
parser.add_argument('--eval-batch-size', type=int, default=2048, metavar='N',
                    help='minimal batch size for evaluation (default: 2048)')
parser.add_argument('--max-iter-num', type=int, default=1000, metavar='N',
                    help='maximal number of main iterations (default: 150)')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--gpu-index', type=int, default=0, metavar='N')


parser.add_argument('--sparse', action='store_true', default=False,
                    help='Sparse env')
parser.add_argument('--disc', action='store_true', default=False,
                    help='disc env')
parser.add_argument('--save-model', action='store_true', default=False,
                    help='saves model')
parser.add_argument('--save-reward', type=float, default=-50, metavar='G',
                    help='Save reward if log_eval is above this value')
parser.add_argument('--episode-length', type=int, default=1000, metavar='N',
                    help='Episode Length')
parser.add_argument('--final', action='store_true', default=False,
                    help='Final')
parser.add_argument('--nn-param', nargs='+', type=int,default=[128,128])
parser.add_argument('--expert', action='store_true', default=False,
                    help='load expwer traj')

args = parser.parse_args()


nn_size = tuple(args.nn_param)
print('NN Size',nn_size)

dtype = torch.float64
torch.set_default_dtype(dtype)
device = torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')
if torch.cuda.is_available():
    torch.cuda.set_device(args.gpu_index)

"""environment"""
if args.disc:
    if args.sparse:
        env = DiscreteDubinGym(is_sparse = True)
        print("Sparse and discrete environment")
        args.env_name = 'DiscreteDubinGymSparse'
    else:
        env = DiscreteDubinGym()
        print("Dense and discrete environment")
        args.env_name = 'DiscreteDubinGymDense'
    eval_env = DiscreteDubinGym()
else:
    if args.sparse:
        env =  ContinuousDubinGym(is_sparse = True)        
        print("Sparse and continous environment")
        args.env_name = 'ContinousDubinGymSparse'
    else:
        env = ContinuousDubinGym()
        print("Dense and continous environment")
        args.env_name = 'ContinousDubinGymDense'
    eval_env = ContinuousDubinGym()



state_dim = env.observation_space.shape[0]
is_disc_action = len(env.action_space.shape) == 0



if args.final is False:
    writer = SummaryWriter('Results/runs/Env_{}/TRPO/TRPO_{}'
        .format(args.env_name,datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))
else:
    writer = SummaryWriter('Results/final_run/Env_{}/TRPO/TRPO_{}'
        .format(args.env_name,datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))



writer.add_text('Env name',str(args.env_name))
writer.add_text('Data type','Un-Filtered')
writer.add_text('NN Size', str(nn_size))
writer.add_text('Min bath size', str(args.min_batch_size))
writer.add_text('Eval bath size', str(args.eval_batch_size))
writer.add_text('Episode length', str(args.episode_length))







"""seeding"""
np.random.seed(args.seed)
torch.manual_seed(args.seed)
env.seed(args.seed)

if args.expert:
    args.model_path = 'learned_models/DiscreteDubinGymDense_trpo.p'


"""define actor and critic"""
if args.model_path is None:
    if is_disc_action:
        policy_net = DiscretePolicy(state_dim, env.action_space.n)
    else:
        policy_net = Policy(state_dim, env.action_space.shape[0], log_std=args.log_std,hidden_size=nn_size)
    value_net = Value(state_dim,hidden_size=nn_size)
else:
    policy_net, _ = pickle.load(open(args.model_path, "rb"))
    value_net = Value(state_dim,hidden_size=nn_size)
policy_net.to(device)
value_net.to(device)

"""create agent"""
agent = Agent(env, policy_net, device, eval_env = eval_env,num_threads=args.num_threads)


def update_params(batch):
    states = torch.from_numpy(np.stack(batch.state)).to(dtype).to(device)
    actions = torch.from_numpy(np.stack(batch.action)).to(dtype).to(device)
    rewards = torch.from_numpy(np.stack(batch.reward)).to(dtype).to(device)
    masks = torch.from_numpy(np.stack(batch.mask)).to(dtype).to(device)
    with torch.no_grad():
        values = value_net(states)

    """get advantage estimation from the trajectories"""
    advantages, returns = estimate_advantages(rewards, masks, values, args.gamma, args.tau, device)

    """perform TRPO update"""
    trpo_step(policy_net, value_net, states, actions, returns, advantages, args.max_kl, args.damping, args.l2_reg)


def main_loop():
    for i_iter in range(args.max_iter_num):
        """generate multiple trajectories that reach the minimum batch_size"""
        batch, log = agent.collect_samples(args.min_batch_size)
        t0 = time.time()
        update_params(batch)
        t1 = time.time()
        """evaluate with determinstic action (remove noise for exploration)"""

        if args.disc:
            _, log_eval = agent.collect_samples(args.eval_batch_size, eval_flag = True,mean_action=False, render=args.render)
        else:
            _, log_eval = agent.collect_samples(args.eval_batch_size, eval_flag = True,mean_action=True, render=args.render)
        t2 = time.time()

        if i_iter % args.log_interval == 0:
            print('{}\tT_sample {:.4f}\tT_update {:.4f}\tT_eval {:.4f}\ttrain_R_min {:.2f}\ttrain_R_max {:.2f}\ttrain_R_avg {:.2f}\teval_R_avg {:.2f}'.format(
                i_iter, log['sample_time'], t1-t0, t2-t1, log['min_reward'], log['max_reward'], log['avg_reward'], log_eval['avg_reward']))

        if (log_eval['avg_reward'] > args.save_reward and args.save_model):
            to_device(torch.device('cpu'), policy_net, value_net)
            pickle.dump((policy_net, value_net),
                        open('learned_models/{}_trpo.p'.format(args.env_name), 'wb'))
            to_device(device, policy_net, value_net)
            print("Done!!!")
            break
        writer.add_scalar('rewards/train_R_avg',log['avg_reward'],i_iter)
        writer.add_scalar('rewards/eval_R_avg',log_eval['avg_reward'],i_iter+1)



        """clean up gpu memory"""
        torch.cuda.empty_cache()

main_loop()
