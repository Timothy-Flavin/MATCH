import numpy as np
import random

class Thompson_Multinomial():
  def __init__(self,alpha, prior_strength=1,experience_strength=1, explore_n=2):
    self.advantageCounts = np.zeros(len(alpha))
    self.ns = np.zeros(len(alpha))
    self.alpha = alpha
    self.prior_strength = prior_strength
    self.exp_strength = experience_strength
    self.epsilon=0 # doesn't do anything here
    self.explore_n = explore_n
    #self.chosen_before = np.zeros(len(alpha))
  
  def choose(self, prior, active):
    #print(active)
    #print(self.ns)
    unexplored = np.logical_and(self.ns<self.explore_n,active>0)
    #print(unexplored)
    #print(np.arange(len(self.ns))[unexplored])
    #input("go>")
    if np.sum(unexplored)>0:
      self.sampled = np.random.choice(np.arange(len(self.ns))[unexplored])
      return self.sampled
    
    #print(f"{self.prior_strength*prior+self.exp_strength*self.advantageCounts} =  {self.prior_strength}*{prior}+{self.exp_strength}*{self.advantageCounts}")
    alphas = self.prior_strength*prior+self.exp_strength*self.advantageCounts
    sampledMean = np.random.dirichlet(np.maximum(alphas,np.ones(alphas.shape[0])/10),1)[0]
    #print(sampledMean)
    self.sampled = np.argmax(sampledMean*active)
    self.active = active
    return self.sampled

  def dist_mode(self, prior):
    if prior==None:
      prior = self.alpha
    return self.prior_strength*prior+self.exp_strength*self.advantageCounts

  def update(self,advantage,sampled:int=None, verbose=0):
    
    if sampled is not None:
      self.sampled = sampled
    self.ns[sampled]+=1 
    norm=(len(self.advantageCounts)-1)
    if verbose>0:
      print(f"Sampled adv: {int(advantage>0)*abs(advantage)}, non sampled adv: {int(advantage<0)*abs(advantage)/norm}")
    self.advantageCounts[self.sampled]+=int(advantage>0)*abs(advantage)
    self.advantageCounts[:self.sampled]+=int(advantage<0)*abs(advantage)/norm
    if self.sampled<self.advantageCounts.shape[0]-1:
      self.advantageCounts[self.sampled+1:]+=int(advantage<0)*abs(advantage)/norm
    if verbose>0:
      print(self.advantageCounts)

  def print_state(self,prior=None):
    print("  State of Thompson multinomial Sampler: ")
    if prior is not None:
      print(f"  Prior: {prior}")
    print(f"  Advantage history: {self.advantageCounts}")
    print(f"  prior weight: {self.prior_strength}, sample weight: {self.exp_strength}")
    if prior is not None:
      print(f"  Sampler arm State with prior: {(prior*self.prior_strength + self.advantageCounts*self.exp_strength)} = ({prior}*{self.prior_strength} + {self.advantageCounts}*{self.exp_strength})")
    else:
      print(f"  Sampler arm State: {(self.advantageCounts*self.exp_strength)} ({self.advantageCounts}*{self.exp_strength})")


class EG_Multinomial():
  def __init__(self, n, Epsilon=0.3, decay=0.95, prior_strength=1,experience_strength=1, learning_rate=0.1, initial_val=0.0):
    self.advantageCounts = np.zeros(n)+initial_val
    self.ns =np.zeros(n)
    self.n = n
    self.StartEpsilon = Epsilon
    self.epsilon = Epsilon
    self.decay = decay
    self.prior_strength = prior_strength
    self.exp_strength = experience_strength
    self.learning_rate = learning_rate
  
  def choose(self, prior, active, n_determ = 2, verbose=0):
    if np.sum(self.ns) < self.n*n_determ:
      if verbose>0:
        print(f"Choosing a never chosen teammate: {self.ns} {np.arange(len(self.ns))[self.ns<2]}")
      self.sampled = np.random.choice(np.arange(len(self.ns))[self.ns<2])
      if verbose > 0:
        print(f"    choice: {self.sampled}, chosen so far: {self.ns}")

    elif random.random()<self.epsilon:
      if verbose>0:
        print(f"eps {self.n}")
      self.sampled = np.random.randint(0,self.n-1)
    else:
      if verbose>0:
        print(f"greedy {self.n}: {(prior*self.prior_strength + self.advantageCounts*self.exp_strength)*active}")
      self.sampled = np.argmax((prior*self.prior_strength + self.advantageCounts*self.exp_strength)*active)
    self.active = active
    return self.sampled

  def update(self,advantage,sampled:int=None,verbose=0):
    if sampled is not None:
      self.sampled = sampled
    if verbose>0:
      print(f"Updating with adv: {advantage}")
    self.ns[self.sampled]+=1
    self.epsilon*=self.decay
    #print(self.epsilon)
    norm=(len(self.advantageCounts)-1)
    if verbose>0:
      print(f"Sampled adv: {self.advantageCounts}, non sampled adv: {int(advantage<0)*abs(advantage)/norm}, n's: {self.ns}")
    self.advantageCounts[self.sampled] = (1-self.learning_rate)*self.advantageCounts[self.sampled]+self.learning_rate*advantage
    #self.advantageCounts[:self.sampled]= (1-self.learning_rate)*self.advantageCounts[:self.sampled] - self.learning_rate*advantage/norm
    #if self.sampled<self.advantageCounts.shape[0]-1:
      #self.advantageCounts[self.sampled+1:]=(1-self.learning_rate)*self.advantageCounts[self.sampled+1:] - self.learning_rate*advantage/norm

  def print_state(self,prior=None):
    print("  State of EG Multinomial Sampler: ")
    if prior is not None:
      print(f"  Prior: {prior}")
    print(f"  Advantage history: {self.advantageCounts}")
    print(f"  prior weight: {self.prior_strength}, sample weight: {self.exp_strength}")
    if prior is not None:
      print(f"  Sampler arm State with prior: {(prior*self.prior_strength + self.advantageCounts*self.exp_strength)} = ({prior}*{self.prior_strength} + {self.advantageCounts}*{self.exp_strength})")
    else:
      print(f"  Sampler arm State: {(self.advantageCounts*self.exp_strength)} ({self.advantageCounts}*{self.exp_strength})")
  
  def dist_mode(self, prior):
    if prior is not None:
      print(f"  Sampler arm State with prior: {(prior*self.prior_strength + self.advantageCounts*self.exp_strength)} = ({prior}*{self.prior_strength} + {self.advantageCounts}*{self.exp_strength})")
    else:
      print(f"  Sampler arm State: {(self.advantageCounts*self.exp_strength)} ({self.advantageCounts}*{self.exp_strength})")
  

class UCB_Multinomial():
  def __init__(self, n, c=1.0, learning_rate=0.1, prior_strength=1, experience_strength=1, initial_val=0.0):
    self.advantageCounts = np.zeros(n)+initial_val
    self.ns =np.zeros(n)
    self.n = n
    self.prior_strength = prior_strength
    self.exp_strength = experience_strength
    self.c = c 
    self.learning_rate = learning_rate
    self.t = 1
    #self.epsilon=1 # does nothgin

  def choose(self, prior, active, n_determ = 2, verbose=0):
    if np.sum(self.ns) < self.n*n_determ:
      if verbose>0:
        print(f"Choosing a never chosen teammate: {self.ns} {np.arange(len(self.ns))[self.ns<2]}")
      self.sampled = np.random.choice(np.arange(len(self.ns))[self.ns<2])
      if verbose > 0:
        print(f"    choice: {self.sampled}, chosen so far: {self.ns}")
    else:
      if verbose>0:
        print(f"greedy {self.n}: {(prior*self.prior_strength + self.advantageCounts*self.exp_strength)*active} + {self.c*np.sqrt(np.log(self.t)/self.ns)}")
      self.sampled = np.argmax((prior*self.prior_strength + self.advantageCounts*self.exp_strength)*active + self.c*np.sqrt(np.log(self.t)/self.ns))
    self.active = active
    return self.sampled

  def update(self,advantage,sampled:int = None,verbose=0):
    if sampled is not None:
      self.sampled = sampled
    self.t+=1
    if verbose>0:
      print(f"Updating with adv: {advantage}")
    self.ns[self.sampled]+=1
    #self.epsilon*=self.decay
    norm=(len(self.advantageCounts)-1)
    if verbose>0:
      print(f"Sampled adv: {self.advantageCounts}, non sampled adv: {int(advantage<0)*abs(advantage)/norm}, n's: {self.ns}")
    self.advantageCounts[self.sampled] = (1-self.learning_rate)*self.advantageCounts[self.sampled]+self.learning_rate*advantage

  def print_state(self,prior=None):
    print("  State of EG Multinomial Sampler: ")
    if prior is not None:
      print(f"  Prior: {prior}")
    print(f"  Advantage history: {self.advantageCounts}")
    print(f"  prior weight: {self.prior_strength}, sample weight: {self.exp_strength}")
    if prior is not None:
      print(f"  Sampler arm State with prior: {(prior*self.prior_strength + self.advantageCounts*self.exp_strength)} = ({prior}*{self.prior_strength} + {self.advantageCounts}*{self.exp_strength})")
    else:
      print(f"  Sampler arm State: {(self.advantageCounts*self.exp_strength)} ({self.advantageCounts}*{self.exp_strength})")

  def dist_mode(self,prior):
    if prior is not None:
      print(f"  Sampler arm State with prior: {(prior*self.prior_strength + self.advantageCounts*self.exp_strength)} = ({prior}*{self.prior_strength} + {self.advantageCounts}*{self.exp_strength})")
    else:
      print(f"  Sampler arm State: {(self.advantageCounts*self.exp_strength)} ({self.advantageCounts}*{self.exp_strength})")
  


class PEEF_Multinomial():
  def __init__(self, n, Epsilon=0.3, decay=0.95, prior_strength=1,experience_strength=1, learning_rate=0.1):
    self.advantageCounts = np.zeros(len(n))
    self.ns = np.ones(len(n))
    self.n = n
    self.StartEpsilon = Epsilon
    self.epsilon = Epsilon
    self.decay = decay
    self.prior_strength = prior_strength
    self.exp_strength = experience_strength
    self.learning_rate = learning_rate
  
  def choose(self, prior, active):
    if random.random()<self.epsilon:
      self.sampled = np.random.randint(0,self.n-1)
    else:
      self.sampled = np.argmax((prior+self.prior_strength + self.advantageCounts*self.exp_strength)*active)
    self.active = active
    return self.sampled

  def update(self,advantage,verbose=0):
    self.epsilon*=self.decay
    norm=(len(self.advantageCounts)-1)
    if verbose>0:
      print(f"Sampled adv: {self.advantageCounts}, non sampled adv: {int(advantage<0)*abs(advantage)/norm}")
    self.advantageCounts[self.sampled] = (1-self.learning_rate)*self.advantageCounts[self.sampled]+self.learning_rate*advantage
    #self.advantageCounts[:self.sampled]= (1-self.learning_rate)*self.advantageCounts[:self.sampled] - self.learning_rate*advantage/norm
    #if self.sampled<self.advantageCounts.shape[0]-1:
      #self.advantageCounts[self.sampled+1:]=(1-self.learning_rate)*self.advantageCounts[self.sampled+1:] - self.learning_rate*advantage/norm


class Thompson_Bernoulli():
  def __init__(self, alpha, prior_strength=1, experience_strength=1):
    self.advantageCounts = np.zeros(2)
    #self.ns = np.zeros(2)
    self.prior_strength = prior_strength
    self.exp_strength = experience_strength
  
  def choose(self, prior, active):
    sampledMean = np.random.beta(prior*self.prior_strength+self.advantageCounts[1]*self.exp_strength,
                             (1-prior)*self.prior_strength+self.advantageCounts[0]*self.exp_strength,1)
    #print(f"Sampled mean: {sampledMean}")
    self.sampled = int(sampledMean>0.5)#np.random.binomial(1,sampledMean,1)
    #print(self.sampled)
    return self.sampled

  def update(self,advantage):
    #self.ns[self.sampled]+=1
    self.advantageCounts[self.sampled]+=int(advantage>0)*abs(advantage)
    self.advantageCounts[1-self.sampled]+=int(advantage<0)*abs(advantage)
  
  def dist_mode(self,prior):
    if prior == None:
      prior = 0.5
    a1 =     prior*self.prior_strength+self.advantageCounts[1]*self.exp_strength
    a2 = (1-prior)*self.prior_strength+self.advantageCounts[0]*self.exp_strength
    return (a2, a1)



def __get_thompson_test__():
  listen_eg = [Thompson_Multinomial(np.zeros(2)+0.1,1,.2),
               Thompson_Multinomial(np.zeros(2)+0.1,1,.2),
               Thompson_Multinomial(np.zeros(2)+0.1,1,.2)]
  target_eg = Thompson_Multinomial(np.zeros(3)+0.1,1,.1)
  return listen_eg, target_eg

def __get_ucb_test__():
  listen_eg = [UCB_Multinomial(2,1,0.2,1,1,0),
               UCB_Multinomial(2,1,0.2,1,1,0),
               UCB_Multinomial(2,1,0.2,1,1,0)]
  target_eg = UCB_Multinomial(3,1,0.2,1,1,0)
  return listen_eg, target_eg

def __get_eg_test__():
  listen_eg = [EG_Multinomial(2,0.5,0.8,1,1,0.2, initial_val = 0),
               EG_Multinomial(2,0.5,0.8,1,1,0.2, initial_val = 0),
               EG_Multinomial(2,0.5,0.8,1,1,0.2, initial_val = 0)]
  target_eg = EG_Multinomial(3,0.5,0.8,1,1,0.2,initial_val=0.5)
  return listen_eg, target_eg



def run_exp(nsteps, func, verbose=0):
  l_correct = np.zeros(shape=(100,3,3)) # trial num, timestep, players, players
  o_correct = np.zeros(shape=(100,3,3))
  # player 1 is the worse player then 2 then 3
  # [advsor,target]
  adv_mat = np.array([
    [0.1,-0.9,-0.9],
    [0.3,0.0,-0.3],
    [0.9,0.5,0.0]
  ])

  p1_listen,p1_target = func()
  p2_listen,p2_target = func()
  p3_listen,p3_target = func()

  players = [[p1_listen,p1_target],
             [p2_listen,p2_target],
             [p3_listen,p3_target],]

  for step in range(nsteps):
    if verbose>0:
      for i,p in enumerate(players):
        print(f"Player {i} state: ")
        for j in range(3):
          print(f"  listen states [{j}] adv: {p[0][j].advantageCounts} ns: {p[0][j].ns}")#print_state(prior = np.zeros(2))}")
        print(f"  player command state: adv: {p[1].advantageCounts} ns: {p[1].ns}")
        p[1].print_state(prior = np.zeros(3))

    commander = random.randint(0,2)# int(input("Commander: "))
    reciever = players[commander][1].choose(prior = np.zeros(3), active = np.ones(3))
    listened = players[reciever][0][commander].choose(prior=np.ones(2)/2, active=np.ones(2))
    reciever_adv = None
    if listened == 1:
      reciever_adv = np.random.normal(adv_mat[commander,reciever],0.25)

    if verbose>0:
      print(f"Commander: {commander}, chose {reciever} who listened: {listened} and got advantage: {reciever_adv}")

    if listened == 1:
      players[reciever][0][commander].update(reciever_adv, verbose=0)
    else:
      players[reciever][0][commander].update(np.random.normal(0,0.05), verbose=0)
    players[commander][1].update(listened-0.5, verbose=0)
    

    listen_correct = np.zeros(shape=(3,3))

    #print("\n\n Listening counts")
    for j in range(3):
      #print("[ ", end="")
      for i,p in enumerate(players):
        #print(p[0][j].advantageCounts,end=", ")
        listen_correct[j,i] += (p[0][j].advantageCounts[0] < p[0][j].advantageCounts[1]) == (adv_mat[j,i]>0)
      #print("]")

    order_correct = np.zeros(shape=(3,3))
    #print('\nSending adv')
    for i,p in enumerate(players):
      #print(f"row: {i}")
      #print(p[1].advantageCounts)
      #print((p[1].advantageCounts - np.mean(p[1].advantageCounts))>0)
      #print(adv_mat[i])
      #print((adv_mat[i]-np.mean(adv_mat[i])) > 0)
      #print("after")
      order_correct[i] += ((p[1].advantageCounts - np.mean(p[1].advantageCounts))>0) == ((adv_mat[i]-np.mean(adv_mat[i])) > 0)
      #print(order_correct)
      #input(((p[1].advantageCounts - np.mean(p[1].advantageCounts))>0) == ((adv_mat[i]-np.mean(adv_mat[i])) > 0))
    #print(np.sum(order_correct) / 9)
    #for i,p in enumerate(players):
      #print(p[1].epsilon)
    
    #print(listen_correct)
    #print(order_correct)
    l_correct[step] = listen_correct
    o_correct[step] = order_correct

    #input("Next line? ")
  return l_correct, o_correct


if __name__ == '__main__':
  print("Testing Samplers")
  import matplotlib.pyplot as plt
  l_correct = np.zeros(shape=(100,100,3,3)) # trial num, timestep, players, players
  o_correct = np.zeros(shape=(100,100,3,3))

  for i in range(100):
    l_correct[i],o_correct[i] = run_exp(100,__get_eg_test__)
  eg_l = np.mean(l_correct, axis=(0,2,3))
  eg_o = np.mean(o_correct, axis=(0,2,3))

  for i in range(100):
    l_correct[i],o_correct[i] = run_exp(100,__get_eg_test__)
  ucb_l = np.mean(l_correct, axis=(0,2,3))
  ucb_o = np.mean(o_correct, axis=(0,2,3))

  for i in range(100):
    l_correct[i],o_correct[i] = run_exp(100,__get_eg_test__)
  th_l = np.mean(l_correct, axis=(0,2,3))
  th_o = np.mean(o_correct, axis=(0,2,3))

  print(np.mean(l_correct, axis=(0,2,3)))
  print(np.mean(o_correct, axis=(0,2,3)))

  plt.plot(eg_l,label="eg listen",ls='dashed',color='green')
  plt.plot(eg_o,label="eg command",color='green')
  plt.plot(ucb_l,label="ucb listen",ls='dashed',color='red')
  plt.plot(ucb_o,label="ucb command",color='red')
  plt.plot(th_l,label="th listen",ls='dashed',color='blue')
  plt.plot(th_o,label="th command",color='blue')
  plt.title("Comparison of Samplers for Environment Abstraction")
  plt.ylabel("Accuracy of advantage direction")
  plt.xlabel("step number")
  plt.grid()
  plt.legend()
  plt.show()