import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from lib.utils import get_living_mask, get_sobel

class NCCAModel2(nn.Module):
    def __init__(self, channel_n, alpha_channel, fire_rate, device=torch.device("cpu")):
        super(NCCAModel2, self).__init__()

        self.device = device
        self.channel_n = channel_n
        self.alpha_channel = alpha_channel

        self.fc0 = nn.Linear(channel_n*8, 256)
        self.fc1 = nn.Linear(256, channel_n, bias=False)
        with torch.no_grad():
            self.fc1.weight.zero_()
        
        self.pool = torch.nn.MaxPool2d(kernel_size=5, padding=2, stride=1)

        self.fire_rate = fire_rate
        self.to(self.device)

    def perceive(self, x, angle):

        def _perceive_with(x, weight):
            size = weight.shape[0]
            padding = (size-1)/2
            conv_weights = torch.from_numpy(weight.astype(np.float32)).to(self.device)
            conv_weights = conv_weights.view(1,1,size,size).repeat(self.channel_n, 1, 1, 1)
            return F.conv2d(x, conv_weights, padding=int(padding), groups=self.channel_n)

        wa_1, wa_2 = get_sobel(3)
        wa_3, wa_4 = get_sobel(5)
        wa_5, wa_6 = get_sobel(7)
        
        wa_1/=np.sum(np.abs(wa_1))
        wa_2/=np.sum(np.abs(wa_2))
        wa_3/=np.sum(np.abs(wa_3))
        wa_4/=np.sum(np.abs(wa_4))
        wa_5/=np.sum(np.abs(wa_5))
        wa_6/=np.sum(np.abs(wa_6))

        y1 = _perceive_with(x, wa_1)
        y2 = _perceive_with(x, wa_2)
        y3 = _perceive_with(x, wa_3)
        y4 = _perceive_with(x, wa_4)
        y5 = _perceive_with(x, wa_5)
        y6 = _perceive_with(x, wa_6)
        y7 = self.pool(x)
        y = torch.cat((x,y1,y2,y3,y4,y5,y6,y7),1)
        return y

    def update(self, x, valid_mask, fire_rate, angle):
        x = x.transpose(1,3)
        pre_life_mask = get_living_mask(x, self.alpha_channel, 3)

        dx = self.perceive(x, angle)
        dx = dx.transpose(1,3)
        
        dx = self.fc0(dx)
        dx = F.relu(dx)
        dx = self.fc1(dx)

        if fire_rate is None:
            fire_rate=self.fire_rate
        stochastic = torch.rand([dx.size(0),dx.size(1),dx.size(2),1])>fire_rate
        stochastic = stochastic.float().to(self.device)
        dx = dx * stochastic
        dx = dx.transpose(1,3)

        x = x+dx

        post_life_mask = get_living_mask(x, self.alpha_channel, 3)
        life_mask = (pre_life_mask & post_life_mask).float()
#         life_mask = (pre_life_mask & post_life_mask & valid_mask.transpose(1,3)).float()
        
        x = x * life_mask
        x = x.transpose(1,3)
        return x * (valid_mask.float())

    def forward(self, x, valid_mask, steps, fire_rate=None, angle=0.0):
        for step in range(steps):
            x = self.update(x, valid_mask, fire_rate, angle)
        return x

def infer(my_model, x, ALPHA_CHANNEL, induction, valid_mask_t, calibration_map, steps):
    history = [x.detach().cpu().numpy(),]
    for _ in range(steps):
        x = my_model(x, valid_mask_t, 1)
        h = torch.softmax(x[..., :ALPHA_CHANNEL], -1)
        t = induction[..., :ALPHA_CHANNEL]
        _delta = t*(h-1)
        delta = _delta * calibration_map * 1.0
        y1 = x[..., :ALPHA_CHANNEL]-delta

        alpha_h = x[..., ALPHA_CHANNEL:(ALPHA_CHANNEL+1)]
        y2 = alpha_h - 2 * (alpha_h-valid_mask_t) * calibration_map * 1.0
        x = torch.cat((y1,y2,x[..., (ALPHA_CHANNEL+1):]), -1)
        history.append(x.detach().cpu().numpy())
    history.reverse()
    return x.detach().cpu().numpy(), history
