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

    def forward(self, J, dsdt, ddsdtda, cost_s, gamma=1, clip=0):
        up = self._get_grid_partials(J, self.periodic, direction='up')
        down = self._get_grid_partials(J, self.periodic, direction='down')

        if clip > 0:
            up = torch.clamp_(up, min=-clip, max=clip)
            down = torch.clamp_(down, min=-clip, max=clip)
            dsdt = torch.clamp(dsdt, min=-clip, max=clip)

        #kink_1 = torch.clamp(-dsdt[2] / ddsdtda[2], min=-self.a_max, max=self.a_max)
        #kink_2 = torch.clamp(-dsdt[3] / ddsdtda[3], min=-self.a_max, max=self.a_max)
        kinks = []
        for i in range(len(dsdt)):
            if (ddsdtda[i] == 0).any():
                continue
            kink = torch.clamp(-dsdt[i] / ddsdtda[i], min=-self.a_max, max=self.a_max)
            kinks.append(kink)
        kinks.append(-self.a_max)
        kinks.append(0)
        kinks.append(self.a_max)

        best_dJdt = None
        best_a = None
        #for action in [kink_1, kink_2, -self.a_max, self.a_max]:
        for action in kinks:
            ddsdt_a = dsdt + ddsdtda * action
            up_dsdt = torch.clamp(ddsdt_a, min=0) 
            down_dsdt = torch.clamp(ddsdt_a, max=0)
            dJdt = torch.sum(up_dsdt * up + down_dsdt * down, dim=0)
            if best_dJdt is None:
                best_dJdt = dJdt
                best_a = action
            else:
                new = dJdt < best_dJdt
                best_dJdt[new] = dJdt[new]
                if isinstance(action, torch.Tensor):
                    best_a[new] = action[new]
                else:
                    best_a[new] = action
            del ddsdt_a
            del up_dsdt
            del down_dsdt
            del dJdt

        best_dJdt *= gamma 
        best_dJdt += cost_s 
        if self.R > 0:
            best_dJdt += self.R * best_a ** 2
        best_dJdt[cost_s < gamma * 1e-5] = 0
        best_a[cost_s < gamma * 1e-5] = 0
        return best_dJdt, best_a

        # bang-bang policy
        ddsdt_a = ddsdtda * self.a_max

        up_dsdt = torch.clamp(dsdt + ddsdt_a, min=0) 
        down_dsdt = torch.clamp(dsdt + ddsdt_a, max=0)
        dJdt_pos = torch.sum(up_dsdt * up + down_dsdt * down, dim=0)

        up_dsdt = torch.clamp(dsdt - ddsdt_a, min=0) 
        down_dsdt = torch.clamp(dsdt - ddsdt_a, max=0)
        dJdt_neg = torch.sum(up_dsdt * up + down_dsdt * down, dim=0)

        a = ((dJdt_neg > dJdt_pos) * 2 - 1) * float(self.a_max)
        #dJdt = torch.minimum(dJdt_pos, dJdt_neg)

        dsdt_a = dsdt + a * ddsdtda
        downwind = dsdt_a < 0
        upwind = dsdt_a > 0

        del up_dsdt
        del down_dsdt
        del dJdt_pos
        del dJdt_neg

        dJds = downwind * down + upwind * up
        dJdt = torch.sum(dsdt_a * dJds, dim=0)

        del dsdt_a
        del dJds

        #assert (dJdt == torch.minimum(dJdt_pos, dJdt_neg)).all()

        if self.R > 0:
            # -dJdt = min_a { (ds/dt + a * d^2s/dtda) * dJ/ds + cost_s + R * a^2 }
            # to minimize the RHS, we take the derivative wrt a and set it equal to 0
            # 0 = d^2s/dtda * dJ/ds + 2R * a
            # since R > 0, solving for a gives the global minimum
            # we use the upwind partials from R=0 to solve for a
            # then project back to the feasible set
            # this is NOT solving the piecewise quadratic for the global minimum!

            new_a = -torch.sum(dJds * ddsdtda, dim=0) / 2 / self.R
            new_a = torch.clamp_(new_a, min=-self.a_max, max=self.a_max) 

            # downwind needs to stay below rho
            # upwind needs to stay above rho
            rho = -dsdt / ddsdtda
            for d in range(4):
                rho[d][ddsdtda[d] == 0] = np.inf
                new_a[downwind[d]] = torch.min(rho[d][downwind[d]], new_a[downwind[d]])
                #assert (a[downwind[d]] <= rho[d][downwind[d]]).all()

                rho[d][ddsdtda[d] == 0] = -np.inf
                new_a[upwind[d]] = torch.max(rho[d][upwind[d]], new_a[upwind[d]])
                #assert (a[upwind[d]] >= rho[d][upwind[d]]).all()
            
            # floating point error
            dsdt_a = dsdt + new_a * ddsdtda
            #assert (dsdt_a[downwind] < 0)).all()
            #assert (dsdt_a[upwind] > 0)).all()
            downwind = dsdt_a < 0
            upwind = dsdt_a > 0
            dJds = downwind * down + upwind * up
            new_dJdt = torch.sum(dsdt_a * dJds, dim=0)

            #assert ((new_dJdt + self.R * new_a ** 2) <= (dJdt + self.R * a ** 2)).all()
            old_cost = dJdt + self.R * a ** 2
            new_cost = new_dJdt + self.R * new_a ** 2
            a[new_cost < old_cost] = new_a[new_cost < old_cost]
            dJdt[new_cost < old_cost] = new_dJdt[new_cost < old_cost]

        dJdt *= gamma 
        dJdt += cost_s + self.R * a ** 2
        return dJdt, a

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

    @classmethod
    def get_state_mesh(cls, state, delta): 
        raise NotImplementedError

    def action_single(self, state, interpolate_fn, eps, gamma, 
            geom=None, lowers=None, uppers=None, steps=None, midpoints=None, 
            clip=0, delta=.01):
        states = self.get_state_mesh(state, delta)

        if geom is not None:
            s = torch.Tensor(states.reshape(states.shape[0], -1))
            state_coords = np.zeros(s.shape)
            for dim in range(states.shape[0]):
                if geom[dim] > 1:
                    state_coord = self.geomcoord(
                            s[dim], lowers[dim], uppers[dim], 
                            steps[dim], midpoints[dim], geom[dim])
                else:
                    state_coord = s[dim]
                state_coords[dim] = state_coord
            state_coords = np.transpose(state_coords)
            J = interpolate_fn(state_coords)
            J = torch.Tensor(J.reshape(states.shape[1:]))
        else:
            J = interpolate_fn(np.transpose(states.reshape(states.shape[0], -1)))
            J = torch.Tensor(J.reshape(states.shape[1:]))

        states = torch.Tensor(states)
        dsdt, ddsdtda = self.dsdt_ddsdtda(states)
        cost_s = self.cost_s(states, eps)

        step_sizes = self.step_sizes
        state_space = self.ss
        self.step_sizes = torch.Tensor([delta] * len(J.shape))
        self.ss = state_space
        dJdt, a = self.forward(J, dsdt, ddsdtda, cost_s, gamma, clip)
        self.step_sizes = step_sizes
        self.state_space = state_space

        while len(dJdt.shape) > 1:
            dJdt = dJdt[1]
            a = a[1]
        return dJdt[1].item(), a[1].item()

class AnalyticValueIter(ValueIter):
    def __init__(self, name, net, dt, lowers, uppers, steps, midpoints, eps, 
            use_cuda=True, geom=None, parent=None, **net_kwargs):
        super().__init__(name, net, dt, 
                lowers, uppers, steps, midpoints, use_cuda, **net_kwargs)
        self.name = name
        self.parent = parent
        self.eps = eps

        self.geom = geom
        if geom is not None:
            grid = []
            for i in range(len(geom)):
                l, u, s, m, g = \
                        self.lowers[i], self.uppers[i], int(self.steps[i]), \
                        self.midpoints[i], self.geom[i]

                if g > 1:
                    sp = self.geomspace(l, u, s, m, g)
                    steps = torch.abs(sp[1:] - sp[:-1])
                    max_res = (u - l) / torch.min(steps)
                    min_res = (u - l) / torch.max(steps)
                    print(f"dim={i} max_res={max_res:.1f} min_res={min_res:.1f}")
                    grid.append(sp)
                else:
                    grid.append(self.linspace(l, u, s, m))
            self.s = torch.stack(torch.meshgrid(grid))
            self.s = self.s.to(self.J.device)
            self.net.state_space = self.s

        self.J = torch.zeros(self.s.shape[1:]).to(self.J.device)
        self.load()

        self.ddsdtda = None

        # for simulating
        lowers = self.lowers.numpy()
        uppers = self.uppers.numpy()
        steps = self.steps.numpy().astype(int)

        self.state_space = UniformMesh(lowers, uppers, steps,
                                       data_dims=2)
        self.action_space = self.linspace(-net.TORQUE_LIMIT, net.TORQUE_LIMIT, 200, 0)

    def load(self):
        try:
            J = np.load(f"outputs/precompute/{self.name}_ctg.npy")
            self.state_space.data[0] = J
            a = np.load(f"outputs/precompute/{self.name}_a.npy")
            self.state_space.data[1] = a
            self.J = torch.Tensor(J)
            self.a = torch.Tensor(a)
            if self.use_cuda:
                self.J = self.J.to(device)
                self.a = self.a.to(device)
        except:
            pass

    def save(self):
        np.save(f"outputs/precompute/{self.name}_ctg", self.J.to(host).detach().numpy())
        np.save(f"outputs/precompute/{self.name}_a", self.a.to(host).detach().numpy())

    def load_parent(self):
        if self.parent is not None:
            [p.load() for p in self.parent]

    def err_fn(self, a):
        return torch.max(torch.abs(a))

    def step(self, J, gamma, clip, step_size=None):
        if step_size is None:
            step_size = len(J)
        dJdt = []
        a = []
        for start in range(0, len(J), step_size):
            end = min(len(J),start + step_size)

            _dJdt, _a = self.net(J[start:end], 
                             self.dsdt[:,start:end], 
                             self.ddsdtda[:,start:end], 
                             self.cost[start:end],
                             gamma=gamma,
                             clip=clip)

            dJdt.append(_dJdt)
            a.append(_a)
        dJdt = torch.cat(dJdt, dim=0)
        a = torch.cat(a, dim=0)
        return dJdt, a

    def run(self, max_iter=1000000, err_tol=.001, use_cuda=True):
        print(f"running {self.name}")
        with torch.no_grad():
            if self.dsdt is None:
                self.dsdt, self.ddsdtda = self.net.dsdt_ddsdtda(self.s)
                
                print(torch.max(self.dsdt))
                self.cost = self.net.cost_s(self.s, eps=self.eps) #/ 10

                print(torch.sum(self.cost == 0).item())
                self.J[self.cost == 0] = 0

            J = self.J
            pbar = tqdm(range(max_iter))
            infobar = tqdm(bar_format='{unit}')

            last_dJdt = None
            #mu = .1
            mu = 0
            #nesterov = True
            nesterov = False

            clip = 0
            gamma = 1 

            for it in pbar: 
                dJdt, a = self.step(J, gamma, clip)
                assert (dJdt[self.cost == 0] == 0).all()
                J_err = self.err_fn(dJdt)

                #dJdt_max = torch.max(torch.abs(dJdt))
                #if dJdt_max > 1:
                #    dJdt /= dJdt_max
                dJdt = torch.clamp_(dJdt, max=1, min=-1)

                if last_dJdt is not None:
                    dJdt = (mu * last_dJdt + self.dt * dJdt) #/ (1 + mu)
                    if nesterov:
                        new_J = J + dJdt
                        new_J -= torch.min(new_J)
                        new_J[self.cost == 0] = 0

                        n_dJdt, a = self.step(new_J, gamma, clip)
                        n_dJdt = torch.clamp_(n_dJdt, max=1, min=-1)
                        dJdt = (mu * last_dJdt + self.dt * n_dJdt) #/ (1 + mu)
                else:
                    dJdt *= self.dt

                J += dJdt
                last_dJdt = dJdt

                J_max = torch.max(J)
                J_min = torch.min(J)

                J -= torch.min(J)
                J[self.cost == 0] = 0
 
                infobar.unit = (#f"a_err: {a_err:.3f}, "
                                f"J_err: {J_err:.4f}, "
                                f"J_min: {J_min:.4f}, "
                                f"J_max: {J_max:.4f}").rjust(infobar.ncols)
                infobar.refresh() 
                if J_err < err_tol:
                    infobar.close()
                    pbar.close()
                    break

                if (it+1) % 10000 == 0:
                    self.J = J.to(host).detach()
                    self.a = a.to(host).detach()
                    self.save()

                if (it+1) % 10000 == 0:
                    cost_fn = lambda x, u: type(self.net).cost_single(x, u, self.eps, 0)
                    policy_fn = lambda s, J_fn: self.net.action_single(s, J_fn, self.eps, 
                            gamma, geom=self.geom, 
                            uppers=self.uppers, lowers=self.lowers, 
                            steps=self.steps, midpoints=self.midpoints)

                    self.simulate(J=J, a=a, cost_fn=cost_fn, policy_fn=policy_fn)

        self.J = J.to(host).detach()
        self.a = a.to(host).detach()
        self.save()

        return it, J_err

    def simulate(self, J=None, a=None, cost_fn=None, policy_fn=None):
        self.load_parent()
        print(f"simulating {self.name}")

        if J is None:
            J = self.J
        try:
            J = J.to(host).detach().numpy()
        except:
            pass
        self.state_space.data[0] = J

        if a is None:
            a = self.a
        try:
            a = a.to(host).detach().numpy()
        except:
            pass
        self.state_space.data[1] = a

        if cost_fn is None:
            cost_fn = lambda x, u: type(self.net).cost_single(x, u, .01, 0)

        if policy_fn is None:
            policy_fn = lambda s, J_fn: self.net.action_single(s, J_fn, .01, 1,
                    geom=self.geom, 
                    uppers=self.uppers,
                    lowers=self.lowers, 
                    steps=self.steps, 
                    midpoints=self.midpoints)

        if self.parent is not None:
            pcost_fn = [lambda x, u: type(p.net).cost_single(x, u, .01, 0) \
                        for p in self.parent]
            pcost_fn = [cost_fn] + pcost_fn
            ppolicy_fn = [lambda s, J_fn: p.net.action_single(s, J_fn, .01, 1, 
                geom=p.geom, 
                uppers=p.uppers.numpy(),
                lowers=p.lowers.numpy(), 
                steps=p.steps.numpy(), 
                midpoints=p.midpoints.numpy()) \
                          for p in self.parent]
            ppolicy_fn = [policy_fn] + ppolicy_fn
            pstate_space = [p.state_space for p in self.parent]
            pstate_space = [self.state_space] + pstate_space
            paction_space = [p.action_space for p in self.parent]
            paction_space = [self.action_space] + paction_space
            puse_policy = [True] * len(self.parent)
            puse_policy = [False] + puse_policy

        # simulate
        procs = []
        for idx, start_state in enumerate(self.env.sim_states):
            #p = sim(f"outputs/videos/{self.name}_mix_{idx}", 
            #        start_state, self.env, 
            #        pstate_space,
            #        action_space=paction_space,
            #        use_policy=puse_policy,
            #        cost_fn=pcost_fn)
            #procs.append(p)

            if self.parent is not None:
                p = sim(f"outputs/videos/p{self.name}_J_{idx}", 
                        start_state, self.env, 
                        pstate_space,
                        action_space=paction_space,
                        cost_fn=pcost_fn,
                        policy_fn=ppolicy_fn)
            else:
                p = sim(f"outputs/videos/{self.name}_J_{idx}", 
                        start_state, self.env, 
                        self.state_space, 
                        action_space=self.action_space, 
                        cost_fn=cost_fn,
                        policy_fn=policy_fn)
            procs.append(p)

            #if self.parent is not None:
            #    p = sim(f"outputs/videos/p{self.name}_hold_{idx}", 
            #            start_state, self.env, 
            #            pstate_space,
            #            action_space=paction_space,
            #            cost_fn=pcost_fn,
            #            policy_fn=[None] * len(self.parent))
            #else:
            #    p = sim(f"outputs/videos/{self.name}_hold_{idx}", 
            #            start_state, self.env, self.state_space, 
            #            action_space=self.action_space, cost_fn=cost_fn,
            #            policy_fn=None)
            #procs.append(p)

            if a is not None:
                if self.parent is not None:
                    p = sim(f"outputs/videos/p{self.name}_a_{idx}", 
                            start_state, self.env, 
                            pstate_space,
                            action_space=paction_space,
                            use_policy=puse_policy,
                            cost_fn=pcost_fn)
                else:
                    p = sim(f"outputs/videos/{self.name}_a_{idx}", 
                            start_state, self.env, 
                            self.state_space, 
                            action_space=self.action_space, 
                            use_policy=True, 
                            cost_fn=cost_fn)
                procs.append(p)

        [p.join() for p in procs]

