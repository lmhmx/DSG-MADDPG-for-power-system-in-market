
import torch
import torch.nn as nn

class Quadratic_Q(nn.Module):
    def __init__(self, num) -> None:
        super().__init__()
        self.Q = nn.parameter.Parameter(torch.zeros([num, num]))
        self.P = nn.parameter.Parameter(torch.zeros([num, 1]))
        self.b = nn.parameter.Parameter(torch.tensor(0.0))
    def forward(self, s) -> None:
        q_1 = ((s@self.Q)*s).sum(dim=1, keepdim=True)
        q_2 = s@self.P
        q = q_1+q_2+self.b
        return q

def mlp(sizes)->nn.Module:
    """
    sizes:
        For example: [input_num, 32, 32, output_num]
    """
    layers = []
    for j in range(len(sizes)-1):
        act = nn.LeakyReLU if(j<len(sizes)-2) else nn.Identity
        layers+= [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

class Q_Functions_for_MADDPG(nn.Module):
    def __init__(self, N, obs_num_s, action_num_s, neighbours, device = "cpu") -> None:
        super().__init__()
        self.N = N
        self.q_s = []
        omega_scale = 500.0
        theta_scale = 1.0
        s_scale = 2.0
        nu_scale = 1.0
        lambda_scale = 500
        mu_scale = 1000
        u_scale = 2.0
        dot_u_scale = 5
        p_scale = 1
        self.device = device
        self.power_scale = 2.0
        self.norm_scale = []
        self.neighbours = neighbours
        for i in range(N):
            # scale_i = [theta_scale, omega_scale, s_scale, nu_scale, nu_scale, lambda_scale, mu_scale, dot_u_scale, u_scale]
            scale_i = [theta_scale, omega_scale, s_scale, nu_scale, nu_scale, lambda_scale, mu_scale, dot_u_scale, u_scale, p_scale]

            for j in self.neighbours[i]:
                scale_i.append(self.power_scale)
            for j in self.neighbours[i]:
                scale_i.append(dot_u_scale)
            for j in self.neighbours[i]:
                scale_i.append(p_scale)
            scale_i.append(u_scale)
            for j in self.neighbours[i]:
                scale_i.append(u_scale)
            self.norm_scale.append(torch.tensor(scale_i, dtype=torch.float32, device=self.device))
        for i in range(N):
            self.q_s.append(mlp([obs_num_s[i]+1+len(self.neighbours[i]), 64, 32, action_num_s[i]]).to(self.device))
            # self.q_s.append(Quadratic_Q(num=obs_num_s[i]+1).to(self.device))
            self.add_module("q_{}".format(i), self.q_s[i])

    def forward(self, s_s, u_s, is_batch = True):
        """
        s_s: list of colomn vector
        u_s: colomn vector
        return:
            q: list of scalar
        """
        if(is_batch == True):
            q = []
            for i in range(self.N):
                # s_a = torch.cat([s_s[i], u_s[:, i:i+1]], dim=1)
                s_a = torch.cat([s_s[i], u_s[:, i:i+1], u_s[:, self.neighbours[i]]], dim=1)
                s_a_scale = self.norm_scale[i]*torch.as_tensor(s_a, device=self.device)
                q.append(self.q_s[i].forward(s_a_scale))
        else:
            raise NotImplementedError("I have not write this. Please check whether you are wrong")
        return q

