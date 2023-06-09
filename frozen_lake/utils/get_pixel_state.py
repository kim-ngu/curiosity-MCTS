import torchvision.transforms as T
import torch
from copy import deepcopy

def get_pixel_state(env):
    render_env = deepcopy(env)
    state = render_env.render()
    state = torch.from_numpy(state).float()
    state = state.permute(2,0,1)
    state = torch.div(state, 255.0)
    transform = T.Resize((96,96))
    state = transform(state)
    return state

