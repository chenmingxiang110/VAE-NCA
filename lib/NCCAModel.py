import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from lib.utils_vis import make_seed, get_living_mask

class NCCAModel(nn.Module):
    def __init__(self, channel_n, alpha_channel, fire_rate,
                 device=torch.device("cpu")):
        super(NCCAModel, self).__init__()

        self.device = device
        self.channel_n = channel_n
        self.alpha_channel = alpha_channel

        self.fc0 = nn.Linear(channel_n*3*2, 128)
        self.fc1 = nn.Linear(128, channel_n, bias=False)
        with torch.no_grad():
            self.fc1.weight.zero_()

        self.fire_rate = fire_rate
        self.stochastic = nn.Dropout2d(p=fire_rate)
        self.to(self.device)

    def perceive(self, x, angle):

        def _perceive_with(x, weight):
            padding = (weight.shape[0]-1)/2
            s_len = weight.shape[0]
            conv_weights = torch.from_numpy(weight.astype(np.float32)).to(self.device)
            conv_weights = conv_weights.view(1,1,s_len,s_len).repeat(self.channel_n, 1, 1, 1)
            return F.conv2d(x, conv_weights, padding=int(padding), groups=self.channel_n)

#         dx = np.outer([1, 2, 1], [-1, 0, 1]) / 8.0  # Sobel filter
#         dy = dx.T
#         c = np.cos(angle*np.pi/180)
#         s = np.sin(angle*np.pi/180)
#         w1 = c*dx-s*dy
#         w2 = s*dx+c*dy
        
        w1 = np.ones([9,9])/81
        w2 = np.ones([5,5])/9

        y1 = _perceive_with(x, w1)
        y2 = _perceive_with(x, w2)
        y = torch.cat((x,y1,y2),1)
        return y
    
    def distill(self, x):
        y = torch.mean(x, [1,2])
        return y

    def update(self, x, valid_mask, fire_rate, angle):
        x = x.transpose(1,3)
        pre_life_mask = get_living_mask(x, self.alpha_channel)

        dx = self.perceive(x, angle)
        dx = dx.transpose(1,3)
        map_feat = self.distill(dx)
        map_feat = map_feat.unsqueeze(1).unsqueeze(1).expand(-1,dx.size(1),dx.size(2),-1)
        dx = torch.cat((dx, map_feat), -1)
        
        dx = self.fc0(dx)
        dx = F.relu(dx)
        dx = self.fc1(dx)

        if fire_rate is None:
            stochastic = self.stochastic
        else:
            stochastic = nn.Dropout2d(p=fire_rate)
        
        dx = dx.transpose(1,3)
        dx = stochastic(dx)

        x = x+dx

        post_life_mask = get_living_mask(x, self.alpha_channel)
        life_mask = (pre_life_mask & post_life_mask).float()
#         life_mask = (pre_life_mask & post_life_mask & valid_mask.transpose(1,3)).float()
        
        x = x * life_mask
        x = x.transpose(1,3)
        return x * (valid_mask.float())

    def forward(self, x, valid_mask, steps, fire_rate=None, angle=0.0):
        for step in range(steps):
            x = self.update(x, valid_mask, fire_rate, angle)
#             seed = make_seed((x.size(1), x.size(2)), x.size(3))
#             seed_t = torch.from_numpy(seed).unsqueeze(0).to(self.device)
#             x = x+seed_t
        return x
