from model import common
import torch.nn as nn


def make_model(args, parent=False):
    return SRResNet(args)

class SRResNet(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(SRResNet, self).__init__()

        n_resblocks = 5
        n_feats = 64
        kernel_size = 3 
        scale = args.scale[0]
        act = nn.PReLU()
        
        self.sub_mean = common.MeanShift(args.rgb_range)
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)

        # define head module
        m_head = [
            nn.Conv2d(3, 64, kernel_size=9, padding=4),
            act
        ]

        # define body module
        m_body = [
            common.ResBlock(
                conv, n_feats, kernel_size, bn=True, act=act, res_scale=args.res_scale
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))
        m_body.append(nn.BatchNorm2d(n_feats))

        # define tail module
        m_tail = [
            common.Upsampler(conv, scale, n_feats, act='prelu'),
            nn.Conv2d(n_feats, 3, kernel_size=9, padding=4)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        x = self.add_mean(x)

        return x 

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

