import torch
import torch.nn as nn
# import torch.nn.functional as F

#--------------------------------------------------

class Context_Skill_Net(nn.Module):
    def __init__(self, obs_size, action_size):
        super(Context_Skill_Net, self).__init__()
        self.name = "Context-Skill"
        self.context_LSTMcell_size = 10
        self.skill_hidden_size = 10
        self.skill_output_size = 5
        self.controller_hidden_size = 20
        
        self.h_prev = torch.zeros(1, self.context_LSTMcell_size, dtype=torch.double)
        self.c_prev = torch.zeros(1, self.context_LSTMcell_size, dtype=torch.double)
        self.Context = nn.LSTMCell(obs_size, self.context_LSTMcell_size)
        
        self.Skill_hidden_linear = nn.Linear(obs_size, self.skill_hidden_size)
        self.Skill_output_linear = nn.Linear(self.skill_hidden_size, self.skill_output_size)
        
        self.Controller_hidden_linear = nn.Linear(self.context_LSTMcell_size+self.skill_output_size, 
                                                  self.controller_hidden_size)
        self.Controller_output_linear = nn.Linear(self.controller_hidden_size, action_size)
        
    def computeTotalNumberOfParameters(self):
        total = 0
        for p in self.parameters():
            total += p.numel()
        return total
        
    def forward(self, obs):
        # Run forward Context module
        h_prev = self.h_prev
        c_prev = self.c_prev
        h_next, c_next = self.Context(obs, (h_prev, c_prev))
        self.h_prev = h_next
        self.c_prev = c_next
        
        # Run forward Skill module
        Skill_hidden_t = self.Skill_hidden_linear(obs)
        Skill_activated_t = torch.tanh(Skill_hidden_t)
        Skill_output_t = self.Skill_output_linear(Skill_activated_t)
        Skill_activated_t = torch.tanh(Skill_output_t)
        
        # Run forward Controller
        Controller_input_t = torch.cat((h_next, Skill_activated_t), 1)
        Controller_hidden_t = self.Controller_hidden_linear(Controller_input_t)
        Controller_activated_t = torch.tanh(Controller_hidden_t)
        Controller_output_t = self.Controller_output_linear(Controller_activated_t)
        output_t = torch.tanh(Controller_output_t)
        
        return output_t

#--------------------------------------------------

class Context_only_Net(nn.Module):
    def __init__(self, obs_size, action_size):
        super(Context_only_Net, self).__init__()
        self.name = "Context-only"
        self.context_LSTMcell_size = 10
        self.controller_hidden_size = 20
        
        self.h_prev = torch.zeros(1, self.context_LSTMcell_size, dtype=torch.double)
        self.c_prev = torch.zeros(1, self.context_LSTMcell_size, dtype=torch.double)
        self.Context = nn.LSTMCell(obs_size, self.context_LSTMcell_size)
        
        self.Controller_hidden_linear = nn.Linear(self.context_LSTMcell_size, self.controller_hidden_size)
        self.Controller_output_linear = nn.Linear(self.controller_hidden_size, action_size)
        
    def computeTotalNumberOfParameters(self):
        total = 0
        for p in self.parameters():
            total += p.numel()
        return total
        
    def forward(self, obs):
        # Run forward Context module
        h_prev = self.h_prev
        c_prev = self.c_prev
        h_next, c_next = self.Context(obs, (h_prev, c_prev))
        self.h_prev = h_next
        self.c_prev = c_next
        
        # Run forward Controller
        Controller_input_t = h_next
        Controller_hidden_t = self.Controller_hidden_linear(Controller_input_t)
        Controller_activated_t = torch.tanh(Controller_hidden_t)
        Controller_output_t = self.Controller_output_linear(Controller_activated_t)
        output_t = torch.tanh(Controller_output_t)
        
        return output_t

#--------------------------------------------------

class Skill_only_Net(nn.Module):
    def __init__(self, obs_size, action_size):
        super(Skill_only_Net, self).__init__()
        self.name = "Skill-only"
        self.skill_hidden_size = 10
        self.skill_output_size = 5
        self.controller_hidden_size = 20
        
        self.Skill_hidden_linear = nn.Linear(obs_size, self.skill_hidden_size)
        self.Skill_output_linear = nn.Linear(self.skill_hidden_size, self.skill_output_size)
        
        self.Controller_hidden_linear = nn.Linear(self.skill_output_size, self.controller_hidden_size)
        self.Controller_output_linear = nn.Linear(self.controller_hidden_size, action_size)
        
    def computeTotalNumberOfParameters(self):
        total = 0
        for p in self.parameters():
            total += p.numel()
        return total
        
    def forward(self, obs):
        # Run forward Skill module
        Skill_hidden_t = self.Skill_hidden_linear(obs)
        Skill_activated_t = torch.tanh(Skill_hidden_t)
        Skill_output_t = self.Skill_output_linear(Skill_activated_t)
        Skill_activated_t = torch.tanh(Skill_output_t)
        
        # Run forward Controller
        Controller_input_t = Skill_activated_t
        Controller_hidden_t = self.Controller_hidden_linear(Controller_input_t)
        Controller_activated_t = torch.tanh(Controller_hidden_t)
        Controller_output_t = self.Controller_output_linear(Controller_activated_t)
        output_t = torch.tanh(Controller_output_t)
        
        return output_t

#--------------------------------------------------
    
    