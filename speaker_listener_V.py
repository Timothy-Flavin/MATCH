import gymnasium as gym
import numpy as np
import random
import torch
import torch.nn.functional as F
from memory_buffer import memory_buffer
import matplotlib.pyplot as plt
from Thompson import Thompson_Multinomial, UCB_Multinomial, EG_Multinomial
#import pettingzoo.mpe.simple_reference_v3
import srv3
#env = make_env(raw_env)
env = srv3.env(render_mode='human')
"""
env.reset()
for agent in env.agent_iter():
  observation, reward, termination, truncation, info = env.last()

  print("------------------")
  print(agent)
  print(observation)
  print(reward)
  print(termination or truncation)
  print(info)
  

  if termination or truncation:
    action = None
  else:
    direction = 4
    say = 1
    action = say*10+direction
    #action = env.action_space(agent).sample() # this is where you would insert your policy
    print(action)
  env.step(action)
  #sinput()
env.close()
"""

def dir_to_action(dir):
  action = np.zeros(5)
  if np.sum(np.square(dir)) < 0.005:
    action[4] = 1
    return action
  if dir[1]>0:
    action[0]+=dir[1]
  else:
    action[2]-=dir[1]
  if dir[0]>0:
    action[1]+=dir[0]
  else:
    action[3]-=dir[0]
  return action

def get_best_dir(obs, r=0.5, noise_std = 0.1, mul=1):
  noise = np.random.normal(np.zeros(2),np.ones(2)*noise_std)
  closest = np.argmin([np.sum(np.square(obs[2:4])), np.sum(np.square(obs[4:6])), np.sum(np.square(obs[6:8]))])
  if random.random()<r:
    best_dir = -0.1 * obs[0:2] + obs[2+2*closest: 4+2*closest] + noise
  else:
    best_dir = -0.1 * obs[0:2] + (obs[2:4] + obs[4:6] + obs[6:8])/3.0 + noise
  act_dir = dir_to_action(mul*best_dir/max(np.sqrt(np.sum(np.square(best_dir))),1.0))
  return act_dir

def get_action_options(obs, action, my_id):
  said = np.zeros([3,5])
  active = np.zeros(3)
  p=0
  for aid in range(3):
    if aid ==my_id:
      said[my_id] = action
      active[my_id] = 1
    else:
      said[aid] = obs[19+6*p:18+6*(p+1)]
      active[aid] = obs[18+6*p]
      p+=1
  return active, said

def get_command_options(obs, action, my_id, noise_std = 0.1, mul=1.0):
  noise = np.random.normal(np.zeros(2),np.ones(2)*noise_std)
  partner_goals = [np.argmax(obs[12:15]),np.argmax(obs[15:18])]
  partner_offsets = np.array([[0.0,0.0],[0.0,0.0]])
  partner_dists = np.zeros(2)
  # partner offset is goal rel pos - partner rel pos
  partner_offsets[0] = obs[2+2*partner_goals[0]: 2+2*partner_goals[0]+2] - obs[8:10] + noise
  partner_dists[0] = np.sqrt(np.sum(np.square(partner_offsets[0])))
  partner_offsets[1] = obs[2+2*partner_goals[1]: 2+2*partner_goals[1]+2] - obs[10:12] + noise
  partner_dists[1] = np.sqrt(np.sum(np.square(partner_offsets[1])))
  
  dist_num = 0
  if partner_dists[1]>partner_dists[0]:
    dist_num = 1
  message_prob = np.zeros(3)
  message_num = np.zeros([3,5])
  po = 0
  for i in range(3):
    if i == my_id:
      message_prob[i] = 1
      message_num[i] = action
      continue

    message_num[i] = dir_to_action(mul*partner_offsets[po]/max(partner_dists[po],1.0))
    if po == dist_num:
      message_prob[i] = 3
    else:
      message_prob[i] = 2
    po+=1
  #print(f"{partner_offsets[dist_num]}/max({partner_dists[dist_num]},{0.001}) = {partner_offsets[dist_num]/max(partner_dists[dist_num],0.001)}")
  return message_prob/6, message_num


def policy2(obs, verbose=0, listener:Thompson_Multinomial=None, commander:Thompson_Multinomial=None, noise=0.1, randcom=False, bad=False): #noise is added to this agents commands and actions
  my_id = int(obs[0])
  obs = obs[1:]
  #`[self_vel[2], all_landmark_rel_positions[6], agent_rel_positions[4], goal_ids[6], communication[2*6]]`
  
  mul = 1
  if bad:
    mul=-1
  my_dir = get_best_dir(obs,noise_std=noise,mul=mul) #gets this agent's choice of direction without following orders
  legal, actions = get_action_options(obs, my_dir, my_id) #gets which agents we can listen to
  act_prior = np.copy(legal)
  if randcom:
    act_prior*=3
    act_prior[my_id] = 1
  act_prior/=max(np.sum(act_prior),0.001)
  message_prior, message_dirs = get_command_options(obs,my_dir, my_id,noise_std=noise, mul=mul)

  chosen_action = None
  chosen_message = None
  anum=0
  if listener is None:
    anum=np.argmax(np.random.multinomial(1,act_prior))
    chosen_action = actions[anum]
  else:
    anum = listener.choose(act_prior,legal)
    chosen_action = actions[anum]

  if bad:
    chosen_action = actions[my_id]
    anum=my_id

  mnum=0
  if commander is None:
    mnum = np.argmax(np.random.multinomial(1,message_prior))
  else:
    mnum = commander.choose(message_prior,np.ones(3))
  chosen_message = np.concatenate([np.zeros(3),message_dirs[mnum]])
  chosen_message[mnum] = 1
  
  if verbose>0:
    print(f"Agent {my_id}\n  Action_prior: {act_prior},\n  Actions: {actions},\n  chosen: {anum} {chosen_action}")
    print(f"Agent {my_id}\n  message_prior: {message_prior},\n  message_dirs: {message_dirs},\n chosen: {mnum} {chosen_message}")
  action = np.concatenate([chosen_action,chosen_message])

  return action, anum, mnum

torch.set_default_dtype(torch.float64)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HIDDEN = 128
LEARNING_RATE = 0.001

# Networks
V_GOOD = torch.nn.Sequential(
    torch.nn.Linear(18, HIDDEN), torch.nn.ReLU(),
    torch.nn.Linear(HIDDEN, HIDDEN), torch.nn.ReLU(),
    torch.nn.Linear(HIDDEN, 1)
).to(DEVICE).float()

OPT_GOOD = torch.optim.Adam(V_GOOD.parameters(), lr = LEARNING_RATE)

def train_V(policy, V, opt, gamma, max_e = 500, mem_buffer = None, num_samples = 32, batch_size = 64, bootstrap=True):
  if mem_buffer == None:
    mem_buffer = memory_buffer(10000,13,31)
  env = srv3.parallel_env()#render_mode='human')
  #print("Playing episode")
  steps = 0
  obs, info = env.reset()
  #print(obs)
  terminated = False
  truncated = False
  obs_ep = []
  act_ep = []
  reward_ep = []
  obs_ep_ = []
  tot_rew = 0
  while env.agents:
    steps += 1
    #action_samp0 = np.array([1,1,0,0,0, 1,0,0, 1,1,0,0,0])
    #action_samp1 = np.array([0,0,1,1,0, 1,0,0, 1,0,0,1,0])
    #action_samp2 = np.array([0,0,0,0,1, 0,1,0, 1,1,0,0,0])
    #print("Agent 0 acting: ")
    ac0,li0,m0 = policy(obs['agent_0'], verbose=0)
    #print("Agent 1 acting: ")
    ac1,li1,m1 = policy(obs['agent_1'], verbose=0)
    #print("Agent 2 acting: ")
    ac2,li2,m2 = policy(obs['agent_2'], verbose=0)
    act = {'agent_0':ac0,'agent_1':ac1,'agent_2':ac2}
    #input("yo")
    obs_, reward, terminated, truncated, info = env.step(act)
    
    tot_rew += reward['agent_0']

    obs_ep    +=[obs['agent_0'],obs['agent_1'],obs['agent_2']]
    obs_ep_   +=[obs_['agent_0'],obs_['agent_1'],obs_['agent_2']]
    reward_ep +=[reward['agent_0'],reward['agent_1'],reward['agent_2']]
    act_ep    +=[act['agent_0'],act['agent_1'],act['agent_2']]
    obs=obs_  
  #print(steps)
  #print(tot_rew)  
  if not bootstrap:
    for ri in range(len(reward_ep)-2,-1,-1):
      reward_ep[ri] += gamma*reward_ep[ri+1]
  #print(reward_ep)
  for i in range(len(reward_ep)-1):
    mem_buffer.save_transition(obs_ep[i],act_ep[i],reward_ep[i],obs_ep_[i],0.0)
  mem_buffer.save_transition(obs_ep[-1],act_ep[-1],reward_ep[-1],obs_ep_[-1],1.0)
  #print(f"Updating Q Network after {steps} steps")
  aloss = 0
  for s in range(num_samples):
    states, actions, rewards, states_, done_ = mem_buffer.sample_memory_to_cuda(batch_size, DEVICE)
    #print(states.shape)
    #print(actions.shape)
    #print(rewards.shape)
    #input("Shapes make sense?")
    target=None
    with torch.no_grad():
      next_actions = torch.clone(actions)
      npstates_ = states_.cpu().numpy()
      #print(npstates_)
      #print(states_.shape)
      #input("Wha")
      for s in range(states_.shape[0]):
        act_,l_,m_ = policy(npstates_[s])
        next_actions[s] = torch.from_numpy(act_).to(DEVICE)
      target = V(states_[:,1:19])
    qval = V(states[:,1:19])
    #print(rewards)
    #print(qval)
    #print("-----------------------------------")
    loss = 0
    if bootstrap:
      loss = F.mse_loss(qval, rewards[:,None] + (1-done_)[:,None]*gamma*target)
    else:
      loss = F.mse_loss(qval, rewards[:,None])# + (1-done_)[:,None]*gamma*target)

    opt.zero_grad()
    aloss+=loss.cpu().item()
    loss.backward()
    opt.step()
  #print(aloss/num_samples)
  return aloss/num_samples, mem_buffer


def test_policies(policy, commanders, listeners, V, gamma, noise=[0.1,0.1,0.1], verbose=1, human=False):
  env = srv3.parallel_env()#render_mode='human')
  if human:
    env = srv3.parallel_env(render_mode='human')
  steps = 0
  obs, info = env.reset()
  terminated = False
  truncated = False
  tot_rew = np.zeros(3)#[0,0,0]
  avg_adv = np.zeros(3)
  ms = [0,1,2]
  litot = np.zeros((3,3))
  prev_reward = None
  el = 0
  while env.agents:
    el+=1
    steps += 1
    ac0,li0,m0 = policy(obs['agent_0'], listener=listeners[0], commander=commanders[0],noise=noise[0], verbose=verbose)
    ac1,li1,m1 = policy(obs['agent_1'], listener=listeners[1], commander=commanders[1],noise=noise[1], verbose=verbose)
    ac2,li2,m2 = policy(obs['agent_2'], listener=listeners[2], commander=commanders[2],noise=noise[2], verbose=verbose,bad=True)
    act = {'agent_0':ac0,'agent_1':ac1,'agent_2':ac2}
    #input("yo")
    lis = [li0,li1,li2]
    #for i in range(3):
      #litot[i,lis[i]] += 1
    if verbose>0:
      print(f"sent {ms}")
      print(f"listened: {lis}")
    
    for c in range(3):
      if commanders[c] is None:
        continue

      n_options = 0
      for c2 in range(3):
        n_options += int(ms[c] == ms[c2])
      if (ms[c]!=c or n_options>1):
        if verbose>0:
          print(f"n_options from {c}: {n_options}")
        if lis[ms[c]] == c: #If it listened to this commander and either it didn't message itself or it has options 
          if verbose>0:
            print(f"Commander {c} was listened to by {ms[c]}")
          commanders[c].update(0.5,sampled=ms[c],verbose=verbose)
          litot[ms[c],c] += 1
        else:
          if verbose>0:
            print(f"Commanders {c} was ignored")
          commanders[c].update(-0.5,sampled=ms[c],verbose=verbose)
      else:
        if verbose>0:
          print(f"Commander {c} only had 1 option")
    qsbefore = []
    with torch.no_grad():      
      qsbefore = np.array([
        V(torch.from_numpy(obs['agent_0'][1:19])[None,:].to(DEVICE)).cpu().numpy()[0,0],
        V(torch.from_numpy(obs['agent_1'][1:19])[None,:].to(DEVICE)).cpu().numpy()[0,0],
        V(torch.from_numpy(obs['agent_2'][1:19])[None,:].to(DEVICE)).cpu().numpy()[0,0],
      ])

    obs_, reward, terminated, truncated, info = env.step(act)
    
    #ac0_,_0,__0 = policy(obs_['agent_0'], listener=None, commander=None,noise=noise[0], verbose=verbose)
    #ac1_,_1,__1 = policy(obs_['agent_1'], listener=None, commander=None,noise=noise[1], verbose=verbose)
    #ac2_,_2,__2 = policy(obs_['agent_2'], listener=None, commander=None,noise=noise[2], verbose=verbose)
    qsafter = []
    with torch.no_grad():
      qsafter = np.array([
        V(torch.from_numpy(obs_['agent_0'][1:19])[None,:].to(DEVICE)).cpu().numpy()[0,0],
        V(torch.from_numpy(obs_['agent_1'][1:19])[None,:].to(DEVICE)).cpu().numpy()[0,0],
        V(torch.from_numpy(obs_['agent_2'][1:19])[None,:].to(DEVICE)).cpu().numpy()[0,0],
      ])
    rews = np.array([reward['agent_0'],reward['agent_1'],reward['agent_2']])
    if prev_reward is None:
      prev_reward = rews
    advs = rews - prev_reward#rews + gamma*qsafter - qsbefore
    prev_reward = rews
    avg_adv += advs
    if verbose>0:
      print(f"Lis : {lis}")
      print(f"advs: {advs} = rews + gamma*qsafter - qsbefore = {rews} + {gamma}*{qsafter} - {qsbefore} ")
    for l in range(3):
      if listeners[l] is None:
        continue
      #print(lis)
      #print(ms)
      n_options=0
      for l2 in range(3):
        if ms[l2]==l:
          n_options+=1
      if lis[l]!=l:
        n_options = 2
      
      if n_options>1:
        #print(f"updateing {l}")
        listeners[l].update(advs[l],sampled = lis[l])
      if verbose>0:
        print(f"Listener {l} adv counts: {listeners[l].advantageCounts}")

    if verbose>0:
      for c in range(3):
        print(f"Commander {c} adv counts: {commanders[c].advantageCounts}")

    ms = [m0,m1,m2]    

    tot_rew += rews
    obs=obs_ 
    if verbose>0:
      print(f"listened tot: {litot}")
      input("Next frame?")
  
  #print(el)
  if verbose>0:
    print(avg_adv) 
    input("huh???")
  return tot_rew
mem=None
losses = []
for i in range(10):
  l, mem = train_V(policy2,V_GOOD,OPT_GOOD,0.99,mem_buffer=mem,bootstrap=False)
  losses.append(l)

#plt.plot(losses)
#plt.show()

def get_samplers():
  commanders = []
  listeners = []
  for i in range(3):
    commanders.append(Thompson_Multinomial(np.ones(3)/3,5,0.05))#UCB_Multinomial(3,0.5,0.1,1,0.1,0))#Thompson_Multinomial(np.ones(3)/3,2,1))
    listeners.append(Thompson_Multinomial(np.ones(3)/3,5,2.0))#UCB_Multinomial(3,0.5,0.1,1,0.1,0))#Thompson_Multinomial(np.ones(3)/3,2,1))
  return commanders, listeners

def get_rand_samplers():
  commanders = []
  listeners = []
  for i in range(3):
    commanders.append(Thompson_Multinomial(np.ones(3)/3,0.001,0.0001))#UCB_Multinomial(3,0.5,0.1,1,0.1,0))#Thompson_Multinomial(np.ones(3)/3,2,1))
    listeners.append(Thompson_Multinomial(np.ones(3)/3,0.001,0.0001))#UCB_Multinomial(3,0.5,0.1,1,0.1,0))#Thompson_Multinomial(np.ones(3)/3,2,1))
  return commanders, listeners

trials=1000
returns_rand = np.zeros([trials,3])
returns_on_policy = np.zeros([trials,3])
returns_learn = np.zeros([trials,3])
verbose=0

commander_adv = np.zeros([3,3])
listener_adv = np.zeros([3,3])

for t in range(trials):
  randers, risteners = get_rand_samplers()
  if t%50 == 0:
    commanders, listeners = get_samplers()
  returns_learn[t] = test_policies(policy2,commanders,listeners,V_GOOD,0.99,verbose=verbose, human=False)
  if verbose>-1 and t<5:
    for ci,c in enumerate(commanders):
      print(f"Commander advantage counts: ")
      c.print_state(prior=np.array([0.1,0.3,0.3]))
      #commander_adv[ci] = commander_adv[ci] + c.advantageCounts
    for li,l in enumerate(listeners):
      print(f"Listener states")
      l.print_state()
      #listener_adv[li] = listener_adv[li] + l.advantageCounts
    #input("Next trial?")
  for ci,c in enumerate(commanders):
    commander_adv[ci] = commander_adv[ci] + c.advantageCounts
  for li,l in enumerate(listeners):
    listener_adv[li] = listener_adv[li] + l.advantageCounts
  returns_on_policy[t] = test_policies(policy2,[None,None,None],[None,None,None],V_GOOD,0.99,verbose=0, human=False)
  returns_rand[t] = test_policies(policy2,randers,risteners,V_GOOD,0.99,verbose=0, human=False)

print(f"Advantages: ")
print(commander_adv)
print(listener_adv)

print("Returns with random communication")
print(np.mean(returns_rand,axis=0))
print(np.std(returns_rand,axis=0))

print("Returns when agents behave on policy")
print(np.mean(returns_on_policy,axis=0))
print(np.std(returns_on_policy,axis=0))

print("Returns over few shot")
print(np.mean(returns_learn[5::50],axis=0))
print(np.std(returns_learn[5::50],axis=0))

print("Returns after few shot")
print(np.mean(returns_learn[49::50],axis=0))
print(np.std(returns_learn[49::50],axis=0))
