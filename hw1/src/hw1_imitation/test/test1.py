from hw1_imitation.model import MSEPolicy, FlowMatchingPolicy
import torch

model = FlowMatchingPolicy(10, 2, 5)

state = torch.randn(10, 10)

print(model.sample_actions(state))



