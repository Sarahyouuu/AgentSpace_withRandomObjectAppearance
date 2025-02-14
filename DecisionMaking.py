import os
import operator
from typing import Tuple
import matplotlib.pyplot as plt
import torch
from Environment import Environment, get_distance_between_locations, get_pairwise_distance
import numpy as np
import math
from math import inf
from torch import nn
from View import plot_env
import pickle


def init_empty_tensor(size: tuple):
    return torch.empty(size, dtype=torch.float64)


class DecisionMaking:
    def __init__(self, params):
        self.params = params
        self.horizon = self.params.TIME_HORIZON
        self.gamma = self.params.GAMMA
        self.mean_goal_returns = dict()
        self.possible_trajectories = []
        self.possible_trajectories_horizon = []
        self.mean_goal_returns_2 = dict()
        self.init_data_tensors() 

    def init_data_tensors(self):
        print("init_data_tensors() called!")  # Debugging

        self.action_step = 0
        self.env_tensor = torch.empty((
            self.params.EPISODE_NUM,
            self.params.EPISODE_ACTION_STEPS,
            self.params.OBJECT_TYPE_NUM + 1,
            self.params.HEIGHT,
            self.params.WIDTH
        ), dtype=torch.float64)

        self.mental_state_tensor = torch.empty((
            self.params.EPISODE_NUM,
            self.params.EPISODE_ACTION_STEPS,
            self.params.OBJECT_TYPE_NUM
        ), dtype=torch.float64)

        self.states_params_tensor = torch.empty((
            self.params.EPISODE_NUM,
            self.params.EPISODE_ACTION_STEPS,
            self.params.OBJECT_TYPE_NUM * 2 + self.params.OBJECT_TYPE_NUM
        ), dtype=torch.float64)

        self.dt_tensor = torch.empty((
            self.params.EPISODE_NUM,
            self.params.EPISODE_ACTION_STEPS
        ), dtype=torch.float64)

        self.episode_step_num = torch.zeros((self.params.EPISODE_NUM, 1), dtype=torch.float64)

        # Correct initialization of few_many_dict
        self.few_many_dict = {'few few': [], 'few many': [], 'many few': [], 'many many': []}
        print("few_many_dict initialized!")  # Debugging

    def generate_behavior(self, few_many=None, episodes_initial_environment=None):
        print("generate_behavior() called!")  # Debugging

        if self.few_many_dict is None:
            self.few_many_dict = {'few few': [], 'few many': [], 'many few': [], 'many many': []}

        for episode in range(5):
            print(f"Episode {episode + 1}: Agent is learning.")
            self.few_many_dict['few few'].append(episode)  # Example data

    def reset_estimates(self, goal_locations):
        self.possible_trajectories = []
        self.possible_trajectories_horizon = []
        self.mean_goal_returns_2 = {tuple(goal): -inf for goal in goal_locations}

    def estimate_reward_of_possible_trajectories(self, environment: Environment):
        for i, trajectory in enumerate(self.possible_trajectories):
            cum_reward = 0
            saved_state = environment.save_state()

            for obj in trajectory:
                goal_map = np.zeros_like(environment._env_map[0])
                goal_map[obj[0], obj[1]] = 1
                next_obs, pred_reward, _, _, _ = environment.step(goal_map)
                environment.set_mental_state(next_obs[1])

                cum_reward += pred_reward * self.gamma

            environment.load_state(saved_state)

            self.mean_goal_returns_2[tuple(trajectory[0])] = max(
                self.mean_goal_returns_2[tuple(trajectory[0])],
                cum_reward / self.possible_trajectories_horizon[i]
            )

    def generate_possible_trajectories(self, agent_location, object_locations, horizon, grabbed_goals):
        if horizon >= self.horizon:
            self.possible_trajectories.append(grabbed_goals)
            self.possible_trajectories_horizon.append(horizon)
            return

        goal_locations = np.concatenate([object_locations, agent_location], axis=0)
        diagonal, straight = get_distance_between_locations(
            agent_location[0, 0], agent_location[0, 1], goal_locations[:, 0], goal_locations[:, 1]
        )

        agent_to_objects_distances = math.sqrt(2) * diagonal + straight
        for obj_id, obj in enumerate(goal_locations):
            dt = np.array(1) if agent_to_objects_distances[obj_id] < 1.4 else agent_to_objects_distances[obj_id]
            new_object_locations = (
                object_locations if agent_to_objects_distances[obj_id] == 0 else
                object_locations[~np.all(object_locations == obj, axis=1)]
            )
            new_grabbed = grabbed_goals + [obj.tolist()]

            self.generate_possible_trajectories(
                agent_location=np.expand_dims(obj, axis=0),
                object_locations=new_object_locations,
                horizon=horizon + dt,
                grabbed_goals=new_grabbed
            )

    def take_action(self, environment: Environment):
        object_locations, agent_location = environment.get_possible_goal_locations()
        self.reset_estimates(np.concatenate([object_locations, agent_location], axis=0))
        self.generate_possible_trajectories(agent_location, object_locations, 0, [])
        self.estimate_reward_of_possible_trajectories(environment)

        best_goal_location = max(self.mean_goal_returns_2.items(), key=operator.itemgetter(1))[0]
        goal_map = np.zeros((self.params.HEIGHT, self.params.WIDTH))
        goal_map[best_goal_location[0], best_goal_location[1]] = 1

        return goal_map