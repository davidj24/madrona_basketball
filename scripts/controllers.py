import torch
from agent import Agent
import pygame

class Controller():
    """Base class for all methods of controlling the agent"""
    def get_action(self, obs_tensor, viewer_instance) -> torch.Tensor:
        """
        Takes the observations for an agent and returns the next action to take
        
        Args:
            obs_tensor: The observation tensor from the environment
            world_idx: the index of the world that the agent to control is in
            agent_idx: The index of the agent in the world to control
        """
        raise NotImplementedError
    


class RLController(Controller):
    """Controller that uses the trained RL policy to select actions"""
    def __init__(self, agent_model: Agent, device: str):
        self.model = agent_model
        self.device = device

    
    def get_action(self, obs_tensor, viewer_instance):
        obs_for_agent = obs_tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            action, _, _ = self.model(obs_for_agent)

        return action.squeeze(0)
    


class HumanController(Controller):
    """Controller for direct interaction with simulation using keyboard inputs"""
    def get_action(self, obs_tensor, viewer_instance):
        return viewer_instance.get_human_action()
        
    

class RulesController(Controller):
    """The Controller based on a hard-coded policy/rules"""
    def get_action(self, obs_tensor, viewer_instance):        
        # Make hard coded  policy later!!!
        has_ball = obs_tensor[30] # 30 is the index in the observation tensor for hasBall
        if has_ball:
            return torch.tensor([0, 0, 0, 0, 0, 1], dtype=torch.int32)  # Shoot
        else:
            return torch.tensor([0, 0, 0, 1, 0, 0], dtype=torch.int32)  # Grab
        

        
        
         