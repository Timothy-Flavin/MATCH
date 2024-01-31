import numpy as np
from Thompson import Thompson_Bernoulli, EG_Multinomial, UCB_Multinomial
from Thompson import Thompson_Multinomial
import matplotlib.pyplot as plt
#Used to generate T and R for the maze

game_map = np.array([
  [1,1,1,1,1,1,0,1],
  [1,1,0,1,0,1,1,1],
  [1,0,1,1,0,1,1,0],
  [1,1,1,0,0,1,0,0],
  [1,0,0,0,1,1,1,1],
  [1,0,1,1,1,0,1,1],
  [1,0,1,0,1,1,0,1],
  [1,1,1,0,1,0,0,2],
])

def clip(pos):
  return [max(min(pos[0],7),0),max(min(pos[1],7),0)]

def get_state(pos):
  if pos[0]==-1 and pos[1]==-1:
    return 64
  return pos[0]*8+pos[1]

def get_pos(state):
  if state==64:
    return [-1,-1]
  else:
    return [int(state/8),state%8]

def take_action(pos, action):
  state_prob_pairs = p(get_state(pos),action)
  prob = np.random.random()
  st=-1
  for s,pr in state_prob_pairs:
    if prob>pr:
      prob-=pr
    else:
      st=s
      break
  return get_pos(st)

def act(pos, action):
  p = [pos[0],pos[1]]
  if action == 0:
    p[0]-=1
  elif action ==1:
    p[1]+=1
  elif action ==2:
    p[0]+=1
  elif action == 3:
    p[1]-=1
  return p

def p(state,action):
  if state == 64 or state==63:
    return zip([64],[1.0])
  r = int(state/8)
  c = state%8
  st = game_map[r,c]
  if st==2:
    return zip([64],[1.0])
  elif st==0:
    return zip([0],[1.0])
    #zip([get_state(clip(act([r,c],action))), 
    #          get_state(clip(act([r,c],(action+1)%4))),
    #          get_state(clip(act([r,c],(action+2)%4))),
    #          get_state(clip(act([r,c],(action+3)%4)))],[0.34,0.22,0.22,0.22])
  return zip([get_state(clip(act([r,c],action))), 
              get_state(clip(act([r,c],(action+1)%4))),
              get_state(clip(act([r,c],(action-1)%4)))],
              [0.8,0.1,0.1])
  
def r(state,action):
  if state>64 or state<0:
    raise Exception("State out of bounds for this game")
  if state == 64:
    return 0
  if game_map[int(state/8),state%8] == 1:
    return -0.05
  elif game_map[int(state/8),state%8] == 0:
    return -1.05 
  else:
    return 10
  
def done(state):
  return state==64

T = np.zeros((4,65,65))
R = np.zeros((4,65))

for s in range(65):
  for a in range(4):
    R[a,s] = r(s,a)
    sa_probs = p(s,a)
    print()
    for sap in sa_probs:
      T[a,s,sap[0]] += sap[1]
      if a==1:
        print(f"{s}, {sap[0]},{sap[1]}")

np.save("T.npy",T)
np.save("R.npy",R)
#print(R)
#print(T)


def valueIteration(initialV, R, T,discount = 0.99, nIterations=np.inf,tolerance=0.01):
  V = initialV
  iterId = 0
  epsilon = 0
  for s in range(V.shape[0]):
    max_a_val = -999999
    for a in range(R.shape[0]):
      if R[a,s] > max_a_val:
        max_a_val = R[a,s]
    V[s] = max_a_val
  iter=0
  while iter<nIterations:
    iterId = iter
    iter+=1
    Vp = np.max(R+discount*np.matmul(T,V),axis=0)
    epsilon = np.sum(np.abs(V-Vp))
    if epsilon<tolerance:
      break
    V=Vp      
  return [V,iterId,epsilon]

def extractPolicy(V, R, T, discount=0.99):
  policy = np.zeros(len(V))
  policy = np.argmax(R+discount*np.matmul(T,V),axis=0)

  return policy 

def evaluatePolicy(policy, R, T, discount):
  Rpi = np.zeros((policy.shape[0],1))
  Tpi = np.zeros((policy.shape[0],policy.shape[0]))
  for s in range(policy.shape[0]):
    Rpi[s,0] = R[policy[s],s]
    Tpi[s,:] = T[policy[s],s]
  V = np.matmul(np.linalg.inv(np.identity(policy.shape[0])-discount*Tpi),Rpi)
  return V
      
def policyIteration(nStates, nActions, initialPolicy, R, T, discount=0.99, nIterations=np.inf):
  policy = initialPolicy
  V = np.zeros(nStates)
  iterId = 0
  aVals = np.zeros((nStates, nActions))
  while iterId < nIterations:
    iterId+=1
    V = evaluatePolicy(policy, R, T, discount)
    #aVals = np.zeros((nStates, nActions))
    aVals = R + discount * np.matmul(T,V)[:,:,0]
    new_policy = np.argmax(aVals,axis=0)
    if np.sum(np.abs(policy-new_policy))==0:
      break
    policy = new_policy
  return [policy,V,iterId, aVals]
   

[policy,V,iterId, Q] = policyIteration(65,4,np.zeros(65,int),R,T,0.99,1000)
print(Q.shape)
print(V)
print(policy)
[V,iterId,epsilon] = valueIteration(np.zeros(65), R, T, 0.99, 1000, 0.01)
policy = extractPolicy(V,R,T)

for i in range(int(len(V)/8)):
  print(V[i*8:(i+1)*8])
  print(policy[i*8:(i+1)*8])


def rand_policy(state):
  return np.random.randint(0,4)
def biased_policy(state):
  if np.random.random()>0.8:
    return policy[state]
  else:
    return rand_policy(state)
  # return np.argmax(np.random.multinomial(1,[0.1,0.4,0.4,0.1]))
def near_perfect(state):
  if np.random.random()>0.3:
    return policy[state]
  else:
    return rand_policy(state)
def perfect(state):
  return policy[state]

class Agent_With_Oracle():
  def __init__(self, policy, value, n_agents, Q=None, sampler="Thompson"):
    self.priors = np.ones(n_agents)/n_agents

    if sampler == "Thompson":
      self.listener = Thompson_Multinomial(self.priors,2.0,0.1)
    elif sampler=="UCB":
      self.listener = UCB_Multinomial(4,0.5,0.1,1,0.1,0)
    else:
      self.listener = EG_Multinomial(4,0.5,decay=0.95,prior_strength=1,experience_strength=0.2)

    if sampler == "Thompson":
      self.oracle = Thompson_Multinomial(self.priors,2.0,0.1)
    elif sampler=="UCB":
      self.oracle = UCB_Multinomial(4,0.5,0.1,1,0.1,0)
    else:
      self.oracle = EG_Multinomial(4,0.5,decay=0.95,prior_strength=1,experience_strength=0.2)

    self.policy = policy
    self.value = value
    self.oracle_num=0
    self.Q_table = Q

  def V(self, state):
    return self.value[state]

  def Q(self, state, action):
    return self.Q_table[action,state]

  def policy_without_oracle(self, state):
    return self.policy(state), 0, None

  def policy_with_oracle(self, state, policies, active):
    leader = self.listener.choose(0.5, active=active)
    self.leader = leader
    chosen_action = 0
    chosen_action = policies[leader].policy(state)
    return chosen_action, leader
  
  def update_sampler(self, adv, verbose=0):
    self.listener.update(adv, verbose)

  def choose_target(self, priors, active):
    if priors == None:
      priors = self.priors
    return self.oracle.choose(priors, active)

  def update_oracle(self, adv, verbose=0):
    self.oracle.update(adv, verbose)

def run_exp(policies, n_trials, n_agents, one_oracle=True, discount=0.99, verbose=0, update_sampler = True, n_shots=3, sampler = "Thompson"):
  if verbose>0:
    print(f"Starting expirient with {n_agents} agents for {n_trials} trials and 'one_oracle' = {one_oracle}")
  cum_rs = np.zeros((n_agents,n_trials))
  
  agents = []
  for a in range(n_agents):
    agents.append(Agent_With_Oracle(policies[a],V,n_agents,Q, sampler=sampler))

  for i in range(n_trials):
    agents_pos=[]
    for a in range(n_agents):
      agents_pos.append([0,0])
    
    if i%n_shots == 0:
      agents = []
      for a in range(n_agents):
        agents.append(Agent_With_Oracle(policies[a],V,n_agents,Q))

    terminated = False
    step=0
    d=1.0
    agents_pos = np.zeros([n_agents,2],int)
    active = np.ones(n_agents)
    while step<500 and not terminated:
      d*=discount
      terminated=True
      step+=1
      command_dirs = np.identity(4) # every agent commands itself
      commands = np.zeros((4,4)) # the other agents
      for oracle_i in range(4):
        actv = np.copy(active)
        actv[oracle_i] = 1
        command = agents[oracle_i].choose_target(None, actv)
        #print(f"agent: {oracle_i} chose {command}")
        command_dirs[command, oracle_i] = 1
      #print(command_dirs)
      if verbose>0:
        print(f"Oracle {oracle_i} chosen with modes {agents[oracle_i].oracle.dist_mode(None)} chooses target: {command}")
      for a in range(n_agents):
        if verbose>0:
          print(f"Agent {a} step: ")
        state = get_state(agents_pos[a])
        action, leader, sampler = None,None,None
        #if a == command:
        action, leader = agents[a].policy_with_oracle(state, agents, command_dirs[a])
        if update_sampler:
          agents[leader].update_oracle(1, verbose)
        for speak in range(4):
          if command_dirs[a][speak] > 0 and speak!=leader:
            agents[speak].update_oracle(-1, verbose)
        #else:
          #action, leader, sampler = agents[a].policy_without_oracle(state)
        if verbose>0:
          print(f"  agent[{a}] commanded: {a==command}, chose policy {leader} taking action: {action} at {agents_pos[a]}")
        agents_pos[a] = take_action(agents_pos[a],action)
        state_ = get_state(agents_pos[a])
        reward = r(state_,action)
        cum_rs[a,i]+= d*reward
        if verbose>0:
          print(f"  Updating sampler with advantage: {reward + discount*agents[a].Q(state_,agents[a].policy(state_)) - agents[a].Q(state,agents[a].policy(state))}")
        if update_sampler and active[a]:
          agents[a].update_sampler(reward + discount*agents[a].V(state_) - agents[a].V(state))
        active[a] = 1-int(done(state_))
        if verbose>0:
          print(f"  Agent [{a}] terminated: {done(state_)}")
      terminated = active.sum() == 0
      if verbose>1:
        input("waiting for input for next frame")
    if verbose>0 or i==n_shots-1:
      print(f"Trial {i} steps {step}")
      for a in agents:
        print(f"  Target modes: {a.oracle.dist_mode(None)}")
        for ai in range(n_agents):
          print(f"    listen modes: {a.listener.dist_mode(None)}")
  return cum_rs

def test_policy(policy, n_trials, debug=False, disc=0.99):
  cum_rs = np.zeros(n_trials)
  for i in range(n_trials):
    terminated = False
    step=0
    agent_pos = [0,0]
    state = 0
    d=disc
    while step<500 and not terminated:
      action = policy(state)      
      agent_pos = take_action(agent_pos,action)#clip(act(agent_pos,action))
      state_ = get_state(agent_pos)
      reward = r(state_,action)
      if debug:
        print(f"state: {state}, state_ {state_}, action: {action}, pos: {agent_pos}, r: {reward}")
      cum_rs[i]+= d*reward
      d*=disc
      terminated = done(state_)
      state=state_
      step+=1
  return cum_rs

N_TRIALS = 5000

print("Testing policies...")
random_rs = test_policy(rand_policy,N_TRIALS)
#print(random_rs)
print(f"Random policy mean: {np.mean(random_rs)}, std: {np.std(random_rs)}")

biased_rs = test_policy(biased_policy,N_TRIALS)
#print(biased_rs)
print(f"Biased policy mean: {np.mean(biased_rs)}, std: {np.std(biased_rs)}")

good_rs = test_policy(near_perfect,N_TRIALS)
#print(good_rs)
print(f"Good policy mean: {np.mean(good_rs)}, std: {np.std(good_rs)}")

perfect_rs = test_policy(perfect,N_TRIALS)
#print(perfect_rs)
print(f"Optimal policy mean: {np.mean(perfect_rs)}, std: {np.std(perfect_rs)}")

print(f"Overall: {(np.mean(random_rs) + np.mean(biased_rs) + np.mean(good_rs) + np.mean(perfect_rs))/4}")

policy_perfs = []
no_message = [random_rs,biased_rs,good_rs,perfect_rs]


SAMPLER = "Thompson"

random_rs = run_exp([rand_policy, biased_policy, near_perfect, perfect],N_TRIALS,4,False,verbose=0, update_sampler=False, sampler=SAMPLER)
print(random_rs.mean(axis=1))
print(random_rs.std(axis=1))
print(random_rs.mean())
rand_message = [random_rs[0],random_rs[1],random_rs[2],random_rs[3]]

update_rs = run_exp([rand_policy, biased_policy, near_perfect, perfect],N_TRIALS,4,False,verbose=0, update_sampler=True, sampler=SAMPLER)
print(update_rs.mean(axis=1))
print(update_rs.std(axis=1))
print(update_rs.mean())
update_message = [update_rs[0],update_rs[1],update_rs[2],update_rs[3]]

#five_rs = run_exp([rand_policy, biased_policy, near_perfect, perfect],N_TRIALS,4,False,verbose=0, update_sampler=True, n_shots=3, sampler=SAMPLER)
#print(five_rs.mean(axis=1))
#print(five_rs.std(axis=1))
#print(five_rs.mean())
#five_message = [five_rs[0],five_rs[1],five_rs[2],five_rs[3]]

comp = []
for i in range(len(update_message)):
  comp+=[no_message[i],rand_message[i],update_message[i]]#,five_message[i]]


colors=['red','red','red','blue','blue','blue','green','green','green']
bp1 = plt.boxplot(comp[0:3],positions=[1,2,3],manage_ticks=True,boxprops=dict(facecolor='firebrick',color="black"), medianprops=dict(color="white"),patch_artist=True)
bp2 = plt.boxplot(comp[3:6],positions=[4,5,6],boxprops=dict(facecolor='darkorange',color="black"), medianprops=dict(color="white"),patch_artist=True)
bp3 = plt.boxplot(comp[6:9],positions=[7,8,9],boxprops=dict(facecolor='navy',color="black"), medianprops=dict(color="white"),patch_artist=True)
bp4 = plt.boxplot(comp[9:12],positions=[10,11,12],boxprops=dict(facecolor='darkgreen',color="black"), medianprops=dict(color="white"),patch_artist=True)
plt.xticks([1,2,3,4,5,6,7,8,9,10,11,12], ['None','Rand','Learn','None','Rand','Learn','None','Rand','Learn','None','Rand','Learn',])
plt.legend([bp1["boxes"][0], bp2["boxes"][0],bp3["boxes"][0], bp4["boxes"][0]], ["Random","Biased","NearPerfect","Optimal"], loc='lower right')
plt.title("Average Individual Rewards Accross Three Expiriments")
plt.xlabel("Trial Communication Type")
plt.ylabel("Cumulative Discounted Individual Reward")
#print(bplot.keys())
#for patch, color in zip(bplot['boxes'], colors):
#  patch.set_facecolor(color)
plt.show()