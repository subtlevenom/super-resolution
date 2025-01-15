from torch import nn
import torch
import torch.optim as optim

class MOD(nn.Module):

    def __init__(self, num_in_ch=3, num_feat=64, num_expert=12):
        super(MOD, self).__init__()
        self.num_expert = num_expert
        self.num_feat = num_feat

        self.FE = nn.Sequential(
            nn.Conv2d(3, num_feat, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(num_feat, num_feat, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_feat),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(num_feat, num_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_feat * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(num_feat * 2, num_feat * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_feat * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(num_feat * 2, num_feat * 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_feat * 4),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(num_feat * 4, num_feat * 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_feat * 4),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(num_feat * 4, num_feat * 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_feat * 4),
            nn.LeakyReLU(0.2, True),
        )

        self.w_gating1 = nn.Parameter(torch.randn(num_feat * 4, self.num_expert))

        m_classifier = [
            nn.Linear(num_feat * 4, num_feat // 2),
            nn.LeakyReLU(0.2, True),
            nn.Linear(num_feat // 2, 1)
        ]
        self.classifier = nn.Sequential(*m_classifier)
        self.classifiers = nn.ModuleList()
        for _ in range(self.num_expert):
            self.classifiers.append(self.classifier)

        self.orthonet = CodeReduction(c_dim=num_feat * 4, feat_hw=1, blocks=self.num_expert)

    def forward(self, x, routing=None):
        feature = self.FE(x)
        B, C, H, W = feature.shape
        feature = feature.view(B, -1, H * W).permute(0, 2, 1)
        if routing == None:
            routing = torch.einsum('bnd,de->bne', feature, self.w_gating1)
            routing = routing.softmax(dim=-1)

        feature, ortho_weight = self.orthonet(feature)
        feature = torch.split(feature, [feature.shape[-1] // self.num_expert] * self.num_expert, dim=-1)

        routing_top = torch.max(routing, dim=-1)[1].unsqueeze(-1).float()
        for i in range(self.num_expert):
            if i == 0:
                output = self.classifiers[0](feature[0])
            else:
                output = torch.where(routing_top == i, self.classifiers[i](feature[i]), output)
        return output, routing, feature, ortho_weight


class CodeReduction(nn.Module):
    def __init__(self, c_dim, feat_hw, blocks=4, prob=False):
        super(CodeReduction, self).__init__()
        self.body = nn.Sequential(
            nn.Linear(c_dim, c_dim * blocks),
            nn.LeakyReLU(0.2, True)
        )
        self.trans = OrthorTransform(c_dim=c_dim * blocks, feat_hw=feat_hw, groups=blocks)
        self.leakyrelu = nn.LeakyReLU(0.2, True)

    def forward(self, feat):
        feat = self.body(feat)
        feat, weight = self.trans(feat)
        feat = self.leakyrelu(feat)
        return feat, weight



class OrthorTransform(nn.Module):
    def __init__(self, c_dim, feat_hw, groups):  
        super(OrthorTransform, self).__init__()

        self.groups = groups
        self.c_dim = c_dim
        self.feat_hw = feat_hw
        self.weight = nn.Parameter(torch.randn(1, feat_hw, c_dim))
        self.opt_orth = optim.Adam(self.parameters(), lr=1e-3, betas=(0.5, 0.99))

    def forward(self, feat):
        pred = feat * self.weight.expand_as(feat)
        return pred, self.weight.view(self.groups, -1)