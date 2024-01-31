import gymnasium as gym
import numpy as np
import random
import torch
import torch.nn.functional as F
from memory_buffer import memory_buffer
import matplotlib.pyplot as plt

#from stable_baselines3 import PPO
#from stable_baselines3.common.env_util import make_vec_env

from Thompson import Thompson_Bernoulli

torch.set_default_dtype(torch.float64)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HIDDEN = 64
LEARNING_RATE = 0.001

def policy1(obs):
  if len(obs.shape)>1:
    #print(obs.shape)
    arr = torch.from_numpy(np.random.randint(0,2,(obs.shape[0],1)).astype(np.float32)).to(DEVICE)
    #print(arr.shape)
    return arr
  else:
    return np.random.randint(0,2,1)[0]

def policy2(obs):
  #print(obs)
  #print(obs.shape)
  if len(obs.shape)>1:
    return (obs[:,3]>0)[:,None]
  else:
    return (obs[3]>0).astype(int)
  
def policy3(obs):
  if len(obs.shape)>1:
    return (obs[:,3]+obs[:,2]>0)[:,None]
  else:
    return (obs[3]+obs[2]>0).astype(int)

def policy4(obs):
  #print(0.05*(obs[0]+obs[1]))
  if len(obs.shape)>1: 
    return (obs[:,3]+obs[:,2] + 0.05*(obs[:,0]+obs[:,1])>0)[:,None]
  else:
    return (obs[3]+obs[2] + 0.05*(obs[0]+obs[1])>0).astype(int)


letter = 'n'
while letter == 'y':
  env = gym.make('CartPole-v1',max_episode_steps=500,render_mode='human')
  obs,info = env.reset()
  truncated = False
  terminated = False
  while not terminated and not truncated:
    #a1 = Q1(torch.cat((f(obs)[None,:],f(np.array([0],np.float32))[None,:]),1)).to('cpu')[0].item()
    #a2 = Q1(torch.cat((f(obs)[None,:],f(np.array([1],np.float32))[None,:]),1)).to('cpu')[0].item()
    action = policy4(obs)
    #print(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
  letter = input('continue? ')


def f(arr):
  return torch.from_numpy(np.array(arr,np.float32)).to(DEVICE)
def l(arr):
  return torch.from_numpy(np.array(arr,int)).to(DEVICE)

def train_Q(policy, Q, opt, gamma, max_e = 500, mem_buffer = None, num_samples = 32, batch_size = 64, eps=0.0):
  if mem_buffer == None:
    mem_buffer = memory_buffer(10000,1,4)
  env = gym.make("CartPole-v1",max_episode_steps=max_e)
  #print("Playing episode")
  steps = 0
  obs,info = env.reset()
  terminated = False
  truncated = False
  obs_ep = []
  act_ep = []
  reward_ep = []
  obs_ep_ = []
  while not (terminated or truncated):
    steps +=1
    act = policy(obs)
    if random.random()<eps:
      act = random.randint(0,1)
    obs_, reward, terminated, truncated, info = env.step(act)
    
    obs_ep+=[obs]
    obs_ep_+=[obs_]
    reward_ep += [reward]
    act_ep += [act]
    obs=obs_  
  #print(steps)  
  if truncated: #Adjust for truncation problem
    reward_ep[-1] = 99
  for ri in range(len(reward_ep)-2,-1,-1):
    reward_ep[ri] += gamma*reward_ep[ri+1]
  #print(reward_ep)
  reward_ep = np.array(reward_ep)
  for i in range(len(reward_ep)-1):
    mem_buffer.save_transition(obs_ep[i],act_ep[i],reward_ep[i],obs_ep_[i],0.0)
  mem_buffer.save_transition(obs_ep[-1],act_ep[-1],reward_ep[-1],obs_ep_[-1],1.0)
  #print(f"Updating Q Network after {steps} steps")
  aloss = 0
  for s in range(num_samples):
    states, actions, rewards, states_, done_ = mem_buffer.sample_memory_to_cuda(batch_size, DEVICE)
    #target=None
    #with torch.no_grad():
      #target = Q(torch.cat([states_,policy(states_)],1))
    qval = Q(torch.cat([states,actions],1))
    #if s==0:
      #print(qval)
      #print(rewards[:,None])
      #print("------------------------------")
    loss = F.mse_loss(qval, rewards[:,None])# + (1-done_)[:,None]*gamma*target)
    opt.zero_grad()
    aloss+=loss.cpu().item()
    loss.backward()
    opt.step()
  #print(aloss/num_samples)
  return aloss/num_samples, mem_buffer

# Networks
Q1 = torch.nn.Sequential(
    torch.nn.Linear(5, HIDDEN), torch.nn.ReLU(),
    torch.nn.Linear(HIDDEN, HIDDEN), torch.nn.ReLU(),
    torch.nn.Linear(HIDDEN, 1)
).to(DEVICE).float()
Q2 = torch.nn.Sequential(
    torch.nn.Linear(5, HIDDEN), torch.nn.ReLU(),
    torch.nn.Linear(HIDDEN, HIDDEN), torch.nn.ReLU(),
    torch.nn.Linear(HIDDEN, 1)
).to(DEVICE).float()
Q3 = torch.nn.Sequential(
    torch.nn.Linear(5, HIDDEN), torch.nn.ReLU(),
    torch.nn.Linear(HIDDEN, HIDDEN), torch.nn.ReLU(),
    torch.nn.Linear(HIDDEN, 1)
).to(DEVICE).float()
Q4 = torch.nn.Sequential(
    torch.nn.Linear(5, HIDDEN), torch.nn.ReLU(),
    torch.nn.Linear(HIDDEN, HIDDEN), torch.nn.ReLU(),
    torch.nn.Linear(HIDDEN, 1)
).to(DEVICE).float()

#for name, param in Q1.named_parameters():
#    print(name, param.size(), param.dtype)

OPT1 = torch.optim.Adam(Q1.parameters(), lr = LEARNING_RATE)
OPT2 = torch.optim.Adam(Q2.parameters(), lr = LEARNING_RATE)
OPT3 = torch.optim.Adam(Q3.parameters(), lr = LEARNING_RATE)
OPT4 = torch.optim.Adam(Q4.parameters(), lr = LEARNING_RATE)
Qs = [Q1,Q2,Q3,Q4]
OPTs = [OPT1,OPT2,OPT3,OPT4]
policy_list = [policy1,policy2,policy3,policy4]


nsteps = 100
for qn in range(4):
  mem=None
  percent = 0
  losses = []
  for i in range(nsteps):
    if i/nsteps >= percent:
      percent+=0.01
      print(f"{i/nsteps}% done")
    l,mem=train_Q(policy_list[qn],Qs[qn],OPTs[qn],0.99,mem_buffer=mem)
    losses+=[l]
  losses = np.array(losses)
  smooth = 10
  sls = np.zeros(int(len(losses)/smooth))
  for i in range(1,int(len(losses)/smooth)):
    sls[i] = min(np.mean(losses[(i-1)*smooth:i*smooth]),100)
  #torch.save(Q,"Policy1_cartpole_Q")
  plt.title("Smoothed Value function loss over time")
  plt.plot(sls)
  plt.xlabel("trial number")
  plt.ylabel("loss")
  plt.grid()
  plt.show()
  plt.savefig("cartpoleLoss")



def Bayes_post_process(Q,policy,oracle,max_e = 500,device='cpu', gamma=0.99, update_tbern = True, verbose=0):
  env = gym.make("CartPole-v1",max_episode_steps=max_e)#, render_mode='human')
  tbern = Thompson_Bernoulli(0.5,1.0,0.1)
  steps = 0
  obs,info = env.reset()
  terminated=False
  truncated = False #torch.from_numpy(obs)[None,:].to(device)
  #print(f"{f(obs)[None,:]}, {f(np.array([policy(obs)],np.float32))[None,:]}")
  #print(f"inp: {torch.cat((f(obs)[None,:],f(np.array([policy(obs)],np.float32))[None,:]),1)}")
  Qold = Q(torch.cat((f(obs)[None,:],f(np.array([policy(obs)],np.float32))[None,:]),1)).to('cpu')[0]
  cum_rew = 0
  while not terminated and not truncated:
    steps += 1
    leader = tbern.choose(0.5)
    if leader==0:
      act = policy(obs)
      #print(f"policy: {act}")
    else:
      act = oracle(obs)
      #print(f"oracle: {act}")
    
    obs_, reward, terminated, truncated, info = env.step(act)
    Qnew = Q(torch.cat((f(obs_)[None,:],f(np.array([policy(obs_)],np.float32))[None,:]),1)).to('cpu')[0]
    adv = (gamma*Qnew.item()+reward) - Qold.item()
    #print(adv.item())
    if update_tbern:
      tbern.update(adv)
    cum_rew += reward
    env.render()
    #print(f"obs: {obs}")
    #print(f"Qold: {Qold.item()}, Qnew: {Qnew.item()}, adv: {adv} adv counts: {tbern.advantageCounts}, {tbern.dist_mode(0.5)}")
    #print(f"Qs: {[Q(torch.cat((f(obs_)[None,:],f(np.array([0],np.float32))[None,:]),1)).to('cpu')[0], Q(torch.cat((f(obs_)[None,:],f(np.array([1],np.float32))[None,:]),1)).to('cpu')[0]]}")
    #input("next?")
    Qold = Qnew
    obs = obs_
  
  if update_tbern:
    print(f"adv counts: {tbern.advantageCounts}, {tbern.dist_mode(0.5)}")

  if verbose>0:
    print(f"Qold: {Qold.item()}, Qnew: {Qnew.item()}, adv: {adv} adv counts: {tbern.advantageCounts}, {tbern.dist_mode(0.5)}")
    print(f"Qs: {[Q(torch.cat((f(obs_)[None,:],f(np.array([0],np.float32))[None,:]),1)).to('cpu')[0], Q(torch.cat((f(obs_)[None,:],f(np.array([1],np.float32))[None,:]),1)).to('cpu')[0]]}")
    print(f"cum_rew: {cum_rew}")
  return cum_rew


def test(Q,policies,device,n_test_trials=100):
  
  rand_scores  = np.zeros((2,len(policies),len(policies)))
  learn_scores = np.zeros((2,len(policies),len(policies)))
  names = []
  for pi,p1 in enumerate(policies):
    for pj,p2 in enumerate(policies):
      print(f"Training with policies {pi+1}, {pj+1}")
      c1 = []
      c2 = []
      for i in range(n_test_trials):
        print(i)
        print(f"Training with policies {pi+1}, {pj+1}")
        #print(Bayes_post_process(Q,p1,p2,device=device))
        c1.append(Bayes_post_process(Qs[pi],p1,p2,device=device))
        c2.append(Bayes_post_process(Qs[pi],p1,p2,device=device,update_tbern=False))
      c1 = np.array(c1)
      c2 = np.array(c2)
      rand_scores[0,pi,pj] = np.mean(c2)
      rand_scores[1,pi,pj] = np.std(c2)

      learn_scores[0,pi,pj] = np.mean(c1)
      learn_scores[1,pi,pj] = np.std(c1)

  np.save(f"CartPoleRandScores{n_test_trials}.npy", rand_scores)
  np.save(f"CartPoleLearnScores{n_test_trials}.npy",learn_scores)

  print("Random scores")
  print(rand_scores)
  print("Learned scores")
  print(learn_scores)


x = []
names = []
c1 = []
c2 = []
test(Qs,[policy1,policy2,policy3,policy4],DEVICE)

