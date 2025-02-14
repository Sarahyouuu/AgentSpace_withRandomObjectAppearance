import numpy as np
from Environment import Environment
from Object import Object

class EnhancedEnvironment(Environment):
    def __init__(self, params, few_many_objects, object_reappears=True):
        super().__init__(params, few_many_objects, object_reappears)
        self.fixed_reward = 10  # Fixed reward for all objects
        self.agent_needs = np.zeros(2)  # Example: Need1, Need2
        self.agent_slopes = np.zeros(2)  # Example: Slope1, Slope2

    def generate_random_objects(self, num_objects_per_type):
        """
        Generate random objects with fixed rewards in the environment.
        :param num_objects_per_type: Number of objects to generate for each type
        """
        self._object_pool.clear()
        for obj_type in range(self.object_type_num):
            for index in range(num_objects_per_type):
                x, y = np.random.randint(0, self.height), np.random.randint(0, self.width)
                new_object = Object(obj_type=obj_type, reward=self.fixed_reward, index=index, visible=True)
                self._env_map[obj_type + 1, x, y] = 1  # Layer 0 for agent, others for objects
                self._object_pool.append(new_object)

    def set_agent_parameters(self, needs, slopes):
        """
        Set the agent's needs and slopes.
        :param needs: A list or array with two elements representing agent needs.
        :param slopes: A list or array with two elements representing agent slopes.
        """
        self.agent_needs = np.array(needs)
        self.agent_slopes = np.array(slopes)

    def update_environment_dynamics(self):
        """
        Updates the environment dynamics based on agent parameters.
        Adjusts mental states based on the agent's needs and slopes.
        """
        self._mental_states += self.agent_needs * self.agent_slopes

    def step(self, action):
        """
        Overrides the base step method to include dynamic updates.
        :param action: The action taken by the agent.
        """
        next_state, reward, done, info = super().step(action)
        self.update_environment_dynamics()
        return next_state, reward, done, info


# Example usage
if __name__ == "__main__":
    from Utils import Utils

    utils = Utils()
    env = EnhancedEnvironment(params=utils.params, few_many_objects=['few', 'many'])
    env.generate_random_objects(num_objects_per_type=5)
    env.set_agent_parameters(needs=[1, -1], slopes=[0.5, 0.3])

    print("Initial Mental States:", env._mental_states)
    env.update_environment_dynamics()
    print("Updated Mental States:", env._mental_states)
