# noqa: D212, D415
"""
# Modified Simple Reference

```{figure} mpe_simple_reference.gif
:width: 140px
:name: simple_reference
```

This environment is modified by Tim Flavin from part of the <a href='..'>MPE environments</a>. Please read that page first for general information.

| Import             | `ModifiedSImpleReference modified_simple_reference` |
|--------------------|--------------------------------------------------|
| Actions            | Continuous                                       |
| Parallel API       | Yes                                              |
| Manual Control     | No                                               |
| Agents             | `agents= [agent_0, agent_1, agent_2]`            |
| Agents             | 3                                                |
| Action Shape       | (13)                                             |
| Action Values      | Continuous(5),Discrete(3),Continuous(5)          |
| Observation Shape  | (2),(6),(4),(9),(6),(4)                          |
| Observation Values | (-inf,inf)                                       |
| State Shape        | (28,)                                            |
| State Values       | (-inf,inf)                                       |


This environment has 3 agents and 3 landmarks of different colors. Each agent wants to get closer to their target landmark, which is known only by the other agents. All agents are simultaneous speakers and listeners.

Locally, the agents are rewarded by their distance to their target landmark. Globally, all agents are rewarded by the average distance of all the agents to their respective landmarks. The relative weight of these rewards is controlled by the `local_ratio` parameter.

Agent observation space: `[self_vel[2], all_landmark_rel_positions[6], agent_rel_positions[4], landmark_ids[9], goal_ids[6], communication[2*3]]`

Agent discrete\continuous action space: `[move up, move right, move down, move left, dont move, comm_target_1, comm_target_2, comm_target_3, comm_dir[5]]`

### Arguments


``` python
simple_reference_v3.env(local_ratio=0.5, max_cycles=25, continuous_actions=False)
```



`local_ratio`:  Weight applied to local reward and global reward. Global reward weight will always be 1 - local reward weight.

`max_cycles`:  number of frames (a step for each agent) until game terminates

`continuous_actions`: Whether agent action spaces are discrete(default) or continuous

"""

import numpy as np
from gymnasium.utils import EzPickle

from pettingzoo.mpe._mpe_utils.core import Agent, Landmark, World
from pettingzoo.mpe._mpe_utils.scenario import BaseScenario
from pettingzoo.utils.conversions import parallel_wrapper_fn

from ModifiedSimpleEnv import SimpleEnv, make_env


class raw_env(SimpleEnv, EzPickle):
    def __init__(
        self, local_ratio=0.5, max_cycles=25, continuous_actions=True, render_mode=None
    ):
        EzPickle.__init__(
            self,
            local_ratio=local_ratio,
            max_cycles=max_cycles,
            continuous_actions=True, #
            render_mode=render_mode,
        )
        assert (
            0.0 <= local_ratio <= 1.0
        ), "local_ratio is a proportion. Must be between 0 and 1."
        scenario = Scenario()
        world = scenario.make_world()
        SimpleEnv.__init__(
            self,
            scenario=scenario,
            world=world,
            render_mode=render_mode,
            max_cycles=max_cycles,
            continuous_actions=True,
            local_ratio=local_ratio,
        )
        self.metadata["name"] = "simple_reference_v3"


env = make_env(raw_env)
parallel_env = parallel_wrapper_fn(env)


class Scenario(BaseScenario):
    def make_world(self):
        self.n_agents = 3
        world = World()
        # set any world properties first
        world.dim_c = self.n_agents+5 # agent target + 5 for direction
        world.collaborative = True  # whether agents share rewards
        # add agents
        world.agents = [Agent() for i in range(self.n_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = f"agent_{i}"
            agent.collide = False
        # add landmarks
        world.landmarks = [Landmark() for i in range(3)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = "landmark %d" % i
            landmark.collide = False
            landmark.movable = False
        return world

    def reset_world(self, world, np_random):
        # random properties for landmarks
        world.landmarks[0].color = np.array([0.75, 0.25, 0.25])
        world.landmarks[1].color = np.array([0.25, 0.75, 0.25])
        world.landmarks[2].color = np.array([0.25, 0.25, 0.75])
        
        # assign goals to agents
        for agent in world.agents:
            agent.goal_a = None
            agent.goal_b = None
            #agent.goal_c = None # TEdit
        # want other agent to go to the goal landmark
        world.agents[0].goal = np_random.choice(world.landmarks)
        world.agents[1].goal = np_random.choice(world.landmarks)
        world.agents[2].goal = np_random.choice(world.landmarks)

        # random properties for agents
        for i, agent in enumerate(world.agents):
          agent.color = agent.goal.color#np.array([0.25, 0.25, 0.25])
          agent.other_goals = []
          for j, agent_2 in enumerate(world.agents):   
            if  i!=j:
              agent.other_goals.append(agent_2.goal.color)
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np_random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(8)# 3 for who to talk to and 2 for the direction
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np_random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    def reward(self, agent, world):
        if agent.goal is None:
            agent_reward = 0.0
        else:
            agent_reward = np.sqrt(
                np.sum(np.square(agent.goal.state.p_pos - agent.state.p_pos))
            )
        return -agent_reward

    def global_reward(self, world):
        all_rewards = sum(self.reward(agent, world) for agent in world.agents)
        return all_rewards / len(world.agents)

    def observation(self, agent, world):
        #`[self_vel[2], all_landmark_rel_positions[6], agent_rel_positions[4], goal_ids[6], communication[2*3]]`
        # goal color
        goal_color = [np.zeros(world.dim_color), np.zeros(world.dim_color)]
        if len(agent.other_goals)>1:
            goal_color[0] = agent.other_goals[0]
            goal_color[1] = agent.other_goals[1]

        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # entity colors
        #entity_color = []
        #for entity in world.landmarks:
        #    entity_color.append(entity.color)
        # communication of all other agents
        my_id = 0
        for i,other in enumerate(world.agents):
            if other is agent:
                my_id = i
        comm = []
        for other in world.agents:
            if other is agent:
                continue
            #print(f"other state.c: {other.state.c}")
            if np.max(other.state.c[0:3]) > 0 and np.argmax(other.state.c[0:3]) == my_id:
                #print(f"concating: {np.ones(1)}, {other.state.c[3:]}")
                comm.append(np.concatenate([np.ones(1),other.state.c[3:]]))
            else:
                #print(f"Other state c: {other.state.c}")
                comm.append(np.zeros(6))
        #print(f"Generated communication for {my_id}: {comm}")
        # Get relative positions of other agents
        agent_pos = []
        for other in world.agents:
            if other is agent:
                continue
            agent_pos.append(other.state.p_pos - agent.state.p_pos)
        return np.concatenate([[my_id]] + [agent.state.p_vel] + entity_pos + agent_pos + goal_color + comm)
