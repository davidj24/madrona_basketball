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
            viewer_instance: The viewer instance for human input
        """
        raise NotImplementedError


class SimpleControllerManager:
    """Simplified controller manager for single-agent training with human override capability"""
    
    def __init__(self, rl_agent: Agent, device: str):
        self.device = device
        self.rl_controller = RLController(rl_agent, device)
        self.human_controller = HumanController()
        
        # Simple flag to determine if human is controlling the current agent
        self.human_control_active = False
        
    def set_human_control(self, active: bool):
        """Enable or disable human control"""
        self.human_control_active = active
        print(f"Human control {'enabled' if active else 'disabled'}")
    
    def get_action(self, obs_tensor: torch.Tensor, viewer_instance=None) -> torch.Tensor:
        """Get action for the current agent"""
        if self.human_control_active and viewer_instance is not None:
            human_action = self.human_controller.get_action(obs_tensor, viewer_instance)
            # Debug: Print when human action is actually used
            if torch.any(human_action != 0):
                print(f"🎮 Using human action: {human_action}")
            return human_action
        else:
            return self.rl_controller.get_action(obs_tensor, viewer_instance)
    
    def is_human_control_active(self) -> bool:
        """Check if human control is currently active"""
        return self.human_control_active


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
        if viewer_instance is not None:
            human_action = viewer_instance.get_human_action()
            # Ensure we return a tensor in the correct format and device
            if isinstance(human_action, torch.Tensor):
                # Make sure it's on the same device as obs_tensor
                return human_action.to(obs_tensor.device)
            else:
                # Convert list to tensor and put on correct device
                tensor_action = torch.tensor(human_action, dtype=torch.int32)
                return tensor_action.to(obs_tensor.device)
        else:
            # Return default action if no viewer, on same device as obs
            default_action = torch.tensor([0, 0, 0, 0, 0, 0], dtype=torch.int32)
            return default_action.to(obs_tensor.device)
        
    

class RulesController(Controller):
    """The Controller based on a hard-coded policy/rules"""
    def get_action(self, obs_tensor, viewer_instance):        
        # Make hard coded  policy later!!!
        has_ball = obs_tensor[30] # 30 is the index in the observation tensor for hasBall
        if has_ball:
            return torch.tensor([0, 0, 0, 0, 0, 1], dtype=torch.int32)  # Shoot
        else:
            return torch.tensor([0, 0, 0, 1, 0, 0], dtype=torch.int32)  # Grab




