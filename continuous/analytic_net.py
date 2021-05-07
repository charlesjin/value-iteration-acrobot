import torch
import torch.nn as nn
import numpy as np
import multiprocessing as mp
from tqdm import tqdm

from utils.sim import sim
from utils.mesh import UniformMesh

from continuous.net import ValueIterNet, ValueIter

host = torch.device("cpu")
device = torch.device("cuda:0") if torch.cuda.is_available() else host
#torch.set_default_tensor_type(torch.DoubleTensor)

class AnalyticValueIterNet(ValueIterNet):
    def __init__(self, step_sizes, periodic, R, a_max, **kwargs):
        super().__init__(step_sizes)
        self.R = R
        self.periodic = periodic
        self.a_max = a_max

    def forward(self, J, dsdt, ddsdtda, cost_s, gamma=1):
        up = self._get_grid_partials(J, self.periodic, direction='up')
        down = self._get_grid_partials(J, self.periodic, direction='down')
        if self.R == 0:
            # bang-bang policy
            ddsdt_a = ddsdtda * self.a_max

            up_dsdt = torch.clamp(dsdt + ddsdt_a, min=0) 
            down_dsdt = torch.clamp(dsdt + ddsdt_a, max=0)
            dJdt_pos = torch.sum(up_dsdt * up + down_dsdt * down, dim=0)

            up_dsdt = torch.clamp(dsdt - ddsdt_a, min=0) 
            down_dsdt = torch.clamp(dsdt - ddsdt_a, max=0)
            dJdt_neg = torch.sum(up_dsdt * up + down_dsdt * down, dim=0)

            a = ((dJdt_neg > dJdt_pos) * 2 - 1) * self.a_max
            dJdt = torch.minimum(dJdt_pos, dJdt_neg) * gamma + cost_s

            dJdt[cost_s < gamma * 1e-5] = 0
            a[cost_s < gamma * 1e-5] = 0
            return dJdt, a

        # RHS = (ds/dt + a * d^2s/dtda) * dJ/dx + cost_s + R * a^2
        # to minimize wrt a, we take the derivative wrt a and set it equal to 0
        # 0 = d^2s/dtda * dJ/dx + 2R * a
        # since R > 0, solving for a gives the global minimum
        # then project a back onto feasible set (which depends on direction)

        dJds = self._get_grid_partials(J, self.periodic, direction='symmetric')
        a = -torch.sum(dJds * ddsdtda, dim=0) / 2 * self.R
        a = torch.clamp_(a, min=-self.a_max, max=self.a_max)
        a[cost_s < gamma * 1e-5] = 0
        dsdt += a * ddsdtda

        up_dsdt = torch.clamp(dsdt, min=0) 
        down_dsdt = torch.clamp(dsdt, max=0)

        cost = torch.sum(up_dsdt * up + down_dsdt * down, dim=0)
        cost *= gamma
        cost += cost_s + self.R * a ** 2
        cost[cost_s < gamma * 1e-5] = 0
        a[cost_s < gamma * 1e-5] = 0

        return cost, a


        #rho = -dsdt / ddsdtda

        #a_up = -up * ddsdtda / (2 * self.R)
        #a_up = torch.clamp_(a_up, min=-self.a_max, max=self.a_max)
        #a_up[a_up < rho] = rho[a_up < rho] # = torch.clamp_(a_up, min=rho)
        #up_cost = torch.sum((dsdt + a_up * ddsdtda) * up, dim=0) + self.R * a_up ** 2
        #up_invalid = rho > self.a_max

        #a_down = -down * ddsdtda / (2 * self.R)
        #a_down = torch.clamp_(a_down, min=-self.a_max, max=self.a_max)
        ##a_down = torch.clamp_(a_down, max=rho)
        #a_down[a_down > rho] = rho[a_down > rho] # = torch.clamp_(a_up, min=rho)
        #down_cost = torch.sum((dsdt + a_down * ddsdtda) * down, dim=0) + self.R * a_down ** 2
        #down_invalid = rho < -self.a_max
        #
        ## at least one of up_valid and down_valid must be valid
        #use_down = down_cost < up_cost
        #up_cost[use_down] = down_cost[use_down]
        #up_cost[up_invalid] = down_cost[up_invalid]
        #up_cost[down_invalid] = up_cost[down_invalid]

        #a_up[use_down] = a_down[use_down]
        #a_up[up_invalid] = a_down[up_invalid]
        #a_up[down_invalid] = a_up[down_invalid]

        #up_cost *= gamma
        #up_cost += cost_s
        #up_cost[cost_s < gamma * 1e-5] = 0
        #a_up[cost_s < gamma * 1e-5] = 0
        return up_cost, a_up

    @classmethod
    def dsdt_ddsdtda(cls, s):
        raise NotImplementedError

    @classmethod
    def cost_s(cls, s, eps):
        raise NotImplementedError

    @classmethod
    def cost_single(cls, s, u, eps, R, add_dims=1):
        dims = len(s)
        s = torch.Tensor(s)
        for _ in range(dims):
            s = torch.unsqueeze(s, dim=-1)
        return cls.cost_s(s, eps).item() + R * u ** 2 

class AnalyticValueIter(ValueIter):
    def __init__(self, net, dt, lowers, uppers, steps, midpoints, use_cuda=True, 
            **net_kwargs):
        super().__init__(net, dt, 
                lowers, uppers, steps, midpoints, use_cuda, **net_kwargs)

        self.J = torch.rand(self.s.shape[1:]).to(self.J.device)
        self.ddsdtda = None

        # for simulating
        lowers = self.lowers.numpy()
        uppers = self.uppers.numpy()
        steps = self.steps.numpy().astype(int)
        self.state_space = UniformMesh(lowers, uppers, steps,
                                       data_dims=2)
        self.action_space = self.linspace(-net.TORQUE_LIMIT, net.TORQUE_LIMIT, 200, 0)

    def err_fn(self, a):
        return torch.max(torch.abs(a))

    def run(self, max_iter=1000000, err_tol=.001, use_cuda=True):
        eps = .5
        with torch.no_grad():
            if self.dsdt is None:
                self.dsdt, self.ddsdtda = self.net.dsdt_ddsdtda(self.s)
                self.cost = self.net.cost_s(self.s, eps=eps) #/ 10

                print(torch.sum(self.cost == 0).item())
                self.J[self.cost == 0] = 0

            J = self.J
            step_size = len(J)
            pbar = tqdm(range(max_iter))
            for it in pbar: 
                gamma = .9
                dJdt = []
                a = []
                for start in range(0, len(J), step_size):
                    end = min(len(J),start + step_size)

                    _dJdt, _a = self.net(J[start:end], 
                                     self.dsdt[:,start:end], 
                                     self.ddsdtda[:,start:end], 
                                     self.cost[start:end],
                                     gamma=gamma)

                    dJdt.append(_dJdt)
                    a.append(_a)
                dJdt = torch.cat(dJdt, dim=0)
                a = torch.cat(a, dim=0)
                assert (dJdt[self.cost == 0] == 0).all()

                J_err = self.err_fn(dJdt)

                #dJdt[J == 0] = 0
                #max_step = torch.min(dJdt / (1e-10 + J))
                #if max_step < -1:
                #    dJdt /= -max_step

                dJdt_max = torch.max(torch.abs(dJdt))
                #dJdt[torch.abs(dJdt) < .0001 * dJdt_max] = 0
                #dJdt[dJdt < .0001 * dJdt_max] = 0
                if dJdt_max > 1:
                    dJdt /= dJdt_max

                #J = dJdt * self.dt + J * (1 - self.dt)
                J += dJdt * self.dt

                #J -= J[10,10,10,10].item()

                J_max = torch.max(J)
                J_min = torch.min(J)

                #J_abs_max = max(-J_min, J_max)
                #if J_abs_max > 100:
                #    #J[abs(J) > 50] = 50.
                #    J = torch.clamp_(J, min=0, max=50)

                J -= torch.min(J)
                J[self.cost == 0] = 0

                pbar.set_postfix_str(#f"a_err: {a_err:.3f}, "
                                     f"J_err: {J_err:.3f}, "
                                     f"J_min: {J_min:.3f}, "
                                     f"J_max: {J_max:.3f}")
                if J_err < err_tol:
                    break

                if (it + 1) % 5000 == 0:
                    #eps = 1. / (1 + it / 100000)
                    #eps = max(eps, .001)
                    #self.cost = self.net.cost(self.s, eps)
                    #print(torch.sum(self.cost == 0).item())

                    np.save(f"outputs/precompute/ctg", 
                               J.to(host).detach().numpy())
                    #a = torch.cat(a, dim=0)
                    self.simulate(J=J, a=a, 
                                  cost_fn=lambda x, u: type(self.net).cost_single(x, u, eps, self.net.R))

        self.J = J.to(host).detach()
        #a = torch.cat(a, dim=0)
        self.a = a.to(host).detach()

    def simulate(self, J=None, a=None, cost_fn=None):
        if J is None:
            J = self.J
        try:
            J = J.to(host).detach().numpy()
        except:
            pass
        self.state_space.data[0] = J

        if a is None:
            a = self.a
        if a is not None:
            try:
                a = a.to(host).detach().numpy()
            except:
                pass
            self.state_space.data[1] = a

        if cost_fn is None:
            cost_fn = lambda x, u: type(self.net).cost_single(x, u, .01, self.net.R)

        # simulate
        procs = []
        for idx, start_state in enumerate(self.env.sim_states):
            p = sim(f"outputs/videos/output_J_{idx}", 
                    start_state, self.env, self.state_space, 
                    action_space=self.action_space, cost_fn=cost_fn)
            procs.append(p)

            if a is not None:
                p = sim(f"outputs/videos/output_a_{idx}", 
                        start_state, self.env, self.state_space, 
                        action_space=self.action_space, use_policy=True, 
                        cost_fn=cost_fn)
                procs.append(p)

        [p.join() for p in procs]

