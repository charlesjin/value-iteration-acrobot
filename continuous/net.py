import torch
import torch.nn as nn
import numpy as np
import multiprocessing as mp
from tqdm import tqdm

from utils.sim import sim
from utils.mesh import UniformMesh

host = torch.device("cpu")
device = torch.device("cuda:0") if torch.cuda.is_available() else host
#torch.set_default_tensor_type(torch.DoubleTensor)

class ValueIterNet(nn.Module):
    def __init__(self, step_sizes, **kwargs):
        super().__init__()
        self.step_sizes = step_sizes

    def _get_grid_partials(self, J, periodic, direction='symmetric'):
        if direction == 'symmetric':
            out = []
            for dim in range(len(J.shape)):
                N = J.shape[dim]
                if dim in periodic: 
                    d0 = torch.narrow(J, dim, 1, 1) - torch.narrow(J, dim, -2, 1)
                else:
                    d0 = torch.zeros(torch.narrow(J, dim, 1, 1).shape).to(J.device)
                d1 = torch.narrow(J, dim, 2, N-2) - torch.narrow(J, dim, 0, N-2)
                out.append((torch.cat((d0, d1, d0), dim=dim) / 2 / self.step_sizes[dim]))
            return torch.stack(out)
        elif direction == 'up':
            up = []
            for dim in range(len(J.shape)):
                N = J.shape[dim]
                if dim in periodic: 
                    d0 = torch.narrow(J, dim, 1, 1) - torch.narrow(J, dim, -1, 1)
                else:
                    d0 = torch.zeros(torch.narrow(J, dim, 1, 1).shape).to(J.device)
                #d0 = torch.narrow(J, dim, 1, 1) - torch.narrow(J, dim, -1, 1)
                d1 = torch.narrow(J, dim, 2, N-2) - torch.narrow(J, dim, 1, N-2)
                up.append((torch.cat((d0, d1, d0), dim=dim) / 2 / self.step_sizes[dim]))
            return torch.stack(up)
        elif direction == 'down':
            down = []
            for dim in range(len(J.shape)):
                N = J.shape[dim]
                if dim in periodic: 
                    d0 = torch.narrow(J, dim, 0, 1) - torch.narrow(J, dim, -2, 1)
                else:
                    d0 = torch.zeros(torch.narrow(J, dim, 1, 1).shape).to(J.device)
                #d0 = torch.narrow(J, dim, 0, 1) - torch.narrow(J, dim, -2, 1)
                d1 = torch.narrow(J, dim, 1, N-2) - torch.narrow(J, dim, 0, N-2)
                down.append((torch.cat((d0, d1, d0), dim=dim) / 2 / self.step_sizes[dim]))
            return torch.stack(down)
        assert False

    def get_grid_partials(self, J, dsdt, periodic, order='first', upwind=True):
        """
        computes partials in all directions, at all grid points
        returns dJ/dx * ds/dt (= -dJ/dt - cost) of the HJB equations

        input J is N1 x N2 x N3 x N4 tensor
        input dsdt is 4 x N1 x N2 x N3 x N4 x N5 tensor
        output is 4 x N1 x N2 x N3 x N4 x N5 tensor

        assume periodic boundary conditions in all dimensions
 
        order can be 
            `first' - first order finite differences approximation
        if upwind is true, use an upwind differencing scheme
        """

        if order == 'first' and not upwind:
            out = self._get_grid_partials(J, periodic, direction='symmetric').unsqueeze(dim=-1)
            return out * dsdt

        elif order == 'first' and upwind:
            up = self._get_grid_partials(J, periodic, direction='up').unsqueeze(dim=-1)
            up_dsdt = torch.clamp(dsdt, min=0)
            ret = up * up_dsdt
            del up
            del up_dsdt

            down = self._get_grid_partials(J, periodic, direction='down').unsqueeze(dim=-1)
            down_dsdt = torch.clamp(dsdt, max=0)
            ret += down * down_dsdt
            return ret

        assert False, f"differencing scheme of order={order} with upwind={upwind} not implemented"

    def forward(self, J, dsdt, cost, gamma=1):
        dJdt_a = torch.sum(
                self.get_grid_partials(J, dsdt, upwind=True), 
                dim=0) * gamma \
            + cost
        dJdt, a = torch.min(dJdt_a, dim=-1)
        return dJdt, a

    @classmethod
    def dsdt(cls, s):
        raise NotImplementedError

    @classmethod
    def cost(cls, s):
        raise NotImplementedError

    @classmethod
    def cost_single(cls, s, eps, add_dims=1):
        s = torch.cat((torch.Tensor(s), torch.zeros(add_dims)))
        dims = len(s)
        for _ in range(dims):
            s = torch.unsqueeze(s, dim=-1)
        return cls.cost(s, eps).item()

class ValueIter(object):
    def __init__(self, net, dt, lowers, uppers, steps, midpoints, use_cuda=True, 
            **net_kwargs):
        self.dt = dt
        self.use_cuda = use_cuda

        assert (steps % 2 == 0).all()
        self.lowers = lowers
        self.uppers = uppers
        self.steps = steps
        self.midpoints = midpoints
        self.step_sizes = (self.uppers - self.lowers) / self.steps

        self.s = torch.stack(
                torch.meshgrid([self.linspace(l, u, int(s), m) \
                        for l, u, s, m in zip(self.lowers, self.uppers, self.steps, 
                                           self.midpoints)]))

        self.net = net(self.step_sizes, **net_kwargs)
        self.env = self.net.env
        self.J = torch.rand(self.s.shape[1:-1])
        self.a = None

        if use_cuda:
            self.s = self.s.to(device)
            self.J = self.J.to(device)
            self.net.to(device)

        self.dsdt = None
        self.cost = None

        # for simulating
        lowers = self.lowers.numpy()
        uppers = self.uppers.numpy()
        steps = self.steps.numpy().astype(int)
        self.state_space = UniformMesh(lowers[:-1], uppers[:-1], steps[:-1],
                                       data_dims=2)
        self.action_space = self.linspace(lowers[-1], uppers[-1], int(steps[-1]), 
                self.midpoints[-1])

    def linspace(self, l, u, s, midpoint=0):
        s = s // 2 + 1
        return torch.cat((
            torch.linspace(l, midpoint, s)[:-1],
            torch.Tensor([midpoint]),
            torch.linspace(midpoint, u, s)[1:]), dim=0)

    def geomspace(self, l, u, s, midpoint=0, res=100):
        s = s // 2 + 1
        diff = (u - l) / 2

        # steps ranges from 0 to diff
        steps = (np.linspace(1, 1+s*res, s) - 1) * diff / (s*res)

        lower = torch.Tensor(-steps[::-1] - midpoint)
        lower[0] = l

        upper = torch.Tensor(steps + midpoint)
        upper[-1] = u

        return torch.cat((
            lower[:-1],
            torch.Tensor([midpoint]),
            upper[1:]), dim=0)

    def err_fn(self, a):
        return torch.max(torch.abs(a))

    def run(self, max_iter=1000000, err_tol=.001, use_cuda=True):
        eps = .1
        with torch.no_grad():
            if self.dsdt is None:
                self.dsdt = self.net.dsdt(self.s)
                self.cost = self.net.cost(self.s, eps=eps) / 10

                print(torch.sum(self.cost == 0).item())
                #print(np.unravel_index(torch.argmin(self.cost + 0).cpu(), self.cost.shape))
                #self.cost[tuple([int(s//2) for s in self.steps])] = 0
                #self.dsdt[:,tuple([int(s//2) for s in self.steps])] = 0

                self.J[torch.min(self.cost, dim=-1) == 0] = 0

            J = self.J
            a = 0 if self.a is None else self.a
            step_size = len(J)
            pbar = tqdm(range(max_iter))
            for it in pbar: 
                #tau = min(1, .00001 * (1+it))
                gamma = .1
                dJdt = []
                a_new = []
                for start in range(0, len(J), step_size):
                    end = min(len(J),start + step_size)

                    _dJdt, _a = self.net(J[start:end], 
                                        self.dsdt[:,start:end], 
                                        self.cost[start:end],
                                        gamma=gamma)

                    dJdt.append(_dJdt)
                    a_new.append(_a)
                dJdt = torch.cat(dJdt, dim=0)
                J_err = self.err_fn(dJdt)

                #a_new = torch.cat(a_new, dim=0)
                #a_err = self.err_fn(a_new - a)
                #a = a_new
                a = torch.cat(a_new, dim=0)

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
                J[torch.min(self.cost, dim=-1) == 0] = 0

                pbar.set_postfix_str(#f"a_err: {a_err:.3f}, "
                                     f"J_err: {J_err:.3f}, "
                                     f"J_min: {J_min:.3f}, "
                                     f"J_max: {J_max:.3f}")
                if J_err < err_tol:
                    break

                if (it + 1) % 10000 == 0:
                    #eps = 1. / (1 + it / 100000)
                    #eps = max(eps, .001)
                    #self.cost = self.net.cost(self.s, eps)
                    #print(torch.sum(self.cost == 0).item())

                    np.save(f"outputs/precompute/ctg", 
                               J.to(host).detach().numpy())
                    #a = torch.cat(a, dim=0)
                    self.simulate(J=J, a=a, 
                                  cost_fn=lambda x, u: type(self.net).cost_single(x, eps))

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
            cost_fn = lambda x, u: type(self.net).cost_single(x, .01)

        # simulate
        procs = []
        for idx, start_state in enumerate(self.env.sim_states):
            #p = sim(f"outputs/videos/output_J_{idx}", 
            #        start_state, self.env, self.state_space, 
            #        action_space=self.action_space, cost_fn=cost_fn)
            #procs.append(p)

            if a is not None:
                p = sim(f"outputs/videos/output_a_{idx}", 
                        start_state, self.env, self.state_space, 
                        action_space=self.action_space, use_policy=True, 
                        cost_fn=cost_fn)
                procs.append(p)

            #p = sim(f"outputs/videos/natural_{idx}", 
            #        start_state, self.env)
            #procs.append(p)

        [p.join() for p in procs]

