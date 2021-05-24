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
    def __init__(self, step_sizes=None, state_space=None, **kwargs):
        super().__init__()
        self.step_sizes = step_sizes
        self.ss = state_space
        assert self.step_sizes is not None or self.state_space is not None
        
        self.den = None

    def _unnorm_partial(self, J, periodic, direction, dim):
        if direction == 'up':
            N = J.shape[dim]
            d0 = torch.narrow(J, dim, 1, 1) - torch.narrow(J, dim, 0, 1)
            if periodic: 
                d2 = d0
            else:
                d2 = torch.zeros(torch.narrow(J, dim, 1, 1).shape).to(J.device)
            d1 = torch.narrow(J, dim, 2, N-2) - torch.narrow(J, dim, 1, N-2)

            return torch.cat((d0, d1, d2), dim=dim)
        elif direction == 'down':
            N = J.shape[dim]
            d2 = torch.narrow(J, dim, -1, 1) - torch.narrow(J, dim, -2, 1)
            if periodic: 
                d0 = d2
            else:
                d0 = torch.zeros(torch.narrow(J, dim, 1, 1).shape).to(J.device)
            d1 = torch.narrow(J, dim, 1, N-2) - torch.narrow(J, dim, 0, N-2)
            return torch.cat((d0, d1, d2), dim=dim)
        else:
            assert False

    def _get_grid_partials(self, J, periodic, direction='symmetric'):
        if self.ss is not None and self.den is None:
            assert isinstance(self.ss, torch.Tensor)
            self.den = []
            for dim in range(len(self.ss)):
                den = torch.abs(
                        self._unnorm_partial(self.ss[dim], dim in periodic, 
                        direction, dim))
                den[den == 0] = 1
                self.den.append(den)

        if direction == 'symmetric':
            up = self._get_grid_partials(J, periodic, 'up')
            down = self._get_grid_partials(J, periodic, 'down')
            return (up + down) / 2.
        else:
            out = []
            for dim in range(len(J.shape)):
                num = self._unnorm_partial(J, dim in periodic, direction, dim)
                if self.den is not None:
                    den = self.den[dim]
                else:
                    den = self.step_sizes[dim]
                out.append(num/den/2)
            return torch.stack(out)
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
    def cost_single(cls, s, eps):
        s = torch.cat((torch.Tensor(s), torch.zeros(1)))
        dims = len(s)
        for _ in range(dims):
            s = torch.unsqueeze(s, dim=-1)
        return cls.cost(s, eps).item()

    @classmethod
    def geomcoord(cls, x, l, u, s, midpoint=0, exp=1.5):
        # converts geometric coordinates into linear coordinates
        s = s // 2 + 1
        assert (x >= l).all()
        assert (x <= u).all()
        c = x - midpoint
        n = torch.log(torch.abs(c) * (exp ** (s-1) - 1) / (u - midpoint) + 1)
        if exp is not None:
            n /= np.log(exp)
        r = (u - l) / 2
        m = r / (s-1) * n * torch.sign(c) + midpoint
        m[m > u] = float(u)
        m[m < l] = float(l)
        return m


class ValueIter(object):
    def __init__(self, name, net, dt, lowers, uppers, steps, midpoints, use_cuda=True, 
            **net_kwargs):
        self.name = name
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
        self.J = torch.zeros(self.s.shape[1:-1])
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
        self.action_space = self.linspace(lowers[-1], uppers[-1], 100, #int(steps[-1]), 
                self.midpoints[-1])

    @classmethod
    def linspace(cls, l, u, s, midpoint=0):
        s = s // 2 + 1
        return torch.cat((
            torch.linspace(l, midpoint, s)[:-1],
            torch.Tensor([midpoint]),
            torch.linspace(midpoint, u, s)[1:]), dim=0)

    @classmethod
    def geomspace(cls, l, u, s, midpoint=0, exp=1.5):
        s = s // 2 + 1
        lower = torch.linspace(-(s-1), 0, s)[:-1]
        upper = torch.linspace(0, s-1, s)[1:]

        lower = exp ** (-lower) - 1
        upper = exp ** upper - 1

        r = (u - l) / 2
        lower = lower / lower[0]
        lower *= -r
        lower += midpoint

        upper = upper / upper[-1]
        upper *= r
        upper += midpoint

        return torch.cat((
            lower,
            torch.Tensor([midpoint]),
            upper), dim=0)

    def err_fn(self, a):
        return torch.max(torch.abs(a))

    def step(self, J, gamma, step_size=None):
        if step_size is None:
            step_size = len(J)
        dJdt = []
        a = []
        for start in range(0, len(J), step_size):
            end = min(len(J),start + step_size)

            _dJdt, _a = self.net(J[start:end], 
                             self.dsdt[:,start:end], 
                             self.cost[start:end],
                             gamma=gamma)

            dJdt.append(_dJdt)
            a.append(_a)
        dJdt = torch.cat(dJdt, dim=0)
        a = torch.cat(a, dim=0)
        return dJdt, a

    def run(self, eps=.01, max_iter=1000000, err_tol=.00001, use_cuda=True):
        with torch.no_grad():
            if self.dsdt is None:
                self.dsdt = self.net.dsdt(self.s)
                self.cost = self.net.cost(self.s, eps=eps)
                print(torch.sum(self.cost == 0).item())

            fixed_points = torch.prod(self.cost, dim=-1) < 1e-6
            self.J[fixed_points] = 0

            J = self.J
            pbar = tqdm(range(max_iter))
            infobar = tqdm(bar_format='{unit}')

            last_dJdt = None
            mu = 0 
            nesterov = False

            gamma = 1
            for it in pbar: 
                dJdt, a = self.step(J, gamma)

                dJdt[fixed_points] = 0
                a[fixed_points] = 0

                J_err = self.err_fn(dJdt)

                dJdt_max = torch.max(torch.abs(dJdt))

                # TODO
                #if dJdt_max > 1:
                #    dJdt /= dJdt_max
                dJdt = torch.clamp_(dJdt, max=1, min=-1)

                if it > 10000 and mu != 0 and last_dJdt is not None:
                    dJdt = (mu * last_dJdt + self.dt * dJdt) / (1 + mu)
                    if nesterov:
                        new_J = J + dJdt
                        new_J -= torch.min(new_J)
                        new_J[fixed_points == 0] = 0

                        n_dJdt, a = self.step(new_J, gamma)
                        n_dJdt = torch.clamp_(n_dJdt, max=1, min=-1)
                        dJdt = (mu * last_dJdt + self.dt * n_dJdt) / (1 + mu)
                else:
                    dJdt *= self.dt

                J += dJdt
                last_dJdt = dJdt

                J_max = torch.max(J)
                J_min = torch.min(J)

                J -= J_min
                J[fixed_points] = 0

                infobar.unit = (#f"a_err: {a_err:.3f}, "
                                f"J_err: {J_err:.4f}, "
                                f"J_min: {J_min:.4f}, "
                                f"J_max: {J_max:.4f}").rjust(infobar.ncols)
                infobar.refresh()
                if J_err < err_tol:
                    infobar.close()
                    pbar.close()
                    break

                if (it + 1) % 1000 == 0:
                    np.save(f"outputs/precompute/{self.name}_ctg", 
                               J.to(host).detach().numpy())
                    np.save(f"outputs/precompute/{self.name}_a", 
                               a.to(host).detach().numpy())

                if (it + 1) % 10000 == 0:
                    cost_fn = lambda x, u: type(self.net).cost_single(x, eps)
                    policy_fn = lambda s, J_fn: self.net.action_single(s, J_fn, eps, gamma)
                    self.simulate(J=J, a=a, 
                            cost_fn=cost_fn, policy_fn=policy_fn, name=name)

        self.J = J.to(host).detach()
        self.a = a.to(host).detach()

        return it, J_err

    def simulate(self, J=None, a=None, cost_fn=None, policy_fn=None):
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

        if policy_fn is None:
            policy_fn = lambda s, J_fn: self.net.action_single(s, J_fn, .01, 1)

        # simulate
        procs = []
        for idx, start_state in enumerate(self.env.sim_states):
            p = sim(f"outputs/videos/{self.name}_J_{idx}", 
                    start_state, self.env, self.state_space, 
                    action_space=self.action_space, cost_fn=cost_fn,
                    policy_fn=policy_fn)
            procs.append(p)

            p = sim(f"outputs/videos/{self.name}_hold_{idx}", 
                    start_state, self.env, self.state_space, 
                    action_space=self.action_space, cost_fn=cost_fn,
                    policy_fn=None)
            procs.append(p)

            if a is not None:
                p = sim(f"outputs/videos/{self.name}_a_{idx}", 
                        start_state, self.env, self.state_space, 
                        action_space=self.action_space, use_policy=True, 
                        cost_fn=cost_fn)
                procs.append(p)

            #p = sim(f"outputs/videos/natural_{idx}", 
            #        start_state, self.env)
            #procs.append(p)

        [p.join() for p in procs]

