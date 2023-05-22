import torch
import torch.nn as nn
import torch.nn.functional as F


class NN_V(nn.Module):

    def __init__(self, i_dim, o_dim):
        super().__init__()
        self.fc1 = nn.Linear(i_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 8)
        self.fc4 = nn.Linear(8, o_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return x


class NN_P(nn.Module):

    def __init__(self, i_dim, o_dim):
        super().__init__()
        self.fc1 = nn.Linear(i_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, o_dim)
        self.fc4 = nn.Softmax(dim=-1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class ac_agent():

    def __init__(self, s, n,  state_num, action_num):  # V(s, w), policy(a\s, theta) ~  NN, s:indice of state
        self.V = NN_V(state_num, 1)
        self.policy = NN_P(state_num, action_num)
        self.s = s
        self.n = n
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.V.to(device)
        self.policy.to(device)


    def take_action(self):
        return torch.distributions.Categorical(
            self.policy(torch.nn.functional.one_hot(torch.tensor(self.s).cuda(), 16).type(torch.float32))
                ).sample()  # the indice of the action

    def update_para(self, s_new_, is_done, r_new_, a_, lr, gamma):
        #print(list(self.V.fc1.parameters()))
        # s_new_ = [s1, s2, ..., sn], so as r_new and a. a is the indices of the action. is_done is the state of s_rew[-1]
        # calculate stationary value function with base-line V for each step ann sum them
        optimizer_V = torch.optim.RMSprop(self.V.parameters(), lr=lr)
        optimizer_p = torch.optim.RMSprop(self.policy.parameters(), lr=lr)
        optimizer_p.zero_grad()
        optimizer_V.zero_grad()
        if is_done:
            svf = 0
        else:
            svf = self.V(torch.nn.functional.one_hot(torch.tensor(s_new_.pop()).cuda(), 16).type(torch.float32))
        loss_V = torch.tensor([0.]).cuda()
        loss_p = torch.tensor([0.]).cuda()
        for i in range(len(r_new_)):
            s_new = s_new_[len(s_new_) - 1 - i]
            r_new = r_new_[len(r_new_) - 1 - i]
            a = a_[len(a_) - 1 - i]
            svf = torch.tensor([r_new]).type(torch.float32).cuda() + gamma * svf
            svf_bl = svf - self.V(torch.nn.functional.one_hot(torch.tensor(s_new).cuda(), 16).type(torch.float32))
            loss_V += svf_bl.pow(2).mean()
            #loss_V.backward(retain_graph=True)
            # uodate paramater of policy
            loss_p -= torch.log(self.policy(torch.nn.functional.one_hot(torch.tensor(s_new).cuda(), 16).type(torch.float32))) @\
                        torch.nn.functional.one_hot(torch.tensor(a).cuda(), 4).type(torch.float32) *\
                            svf_bl.detach().mean()
            #optimizer_p.zero_grad()
            #loss_p.backward(retain_graph=True)
        loss_V.backward()
        loss_p.backward()
        optimizer_V.step()
        optimizer_p.step()
        #print(list(self.V.fc1.parameters()))
