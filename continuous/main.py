from tqdm import tqdm
import torch
import numpy as np
import pickle

from continuous.acrobot_net import AcrobotNet
from continuous.pendulum_net import PendulumNet
from continuous.net import ValueIter

from continuous.analytic_acrobot_net import AnalyticAcrobotNet
from continuous.analytic_pendulum_net import AnalyticPendulumNet
from continuous.analytic_net import AnalyticValueIter

def make_acrobot_vi(use_cuda=True):
    A = AcrobotNet
    lowers = torch.Tensor([
        0, 
        -A.PI, 
        -A.MAX_VEL_1, 
        -A.MAX_VEL_2, 
        -A.TORQUE_LIMIT])
    uppers = torch.Tensor([
        2*A.PI, 
        A.PI, 
        A.MAX_VEL_1, 
        A.MAX_VEL_2, 
        A.TORQUE_LIMIT])
    midpoints = torch.Tensor([
        A.PI,
        0,
        0,
        0,
        0])
    steps = torch.Tensor([20, 20, 20, 20, 4])
    vi = ValueIter(A, .001, 
                   lowers, uppers, steps, midpoints, 
                   use_cuda=use_cuda)
    return vi

def make_analytic_acrobot_vi(use_cuda=True):
    A = AnalyticAcrobotNet

    midpoints = torch.Tensor([
        A.PI,
        0,
        0,
        0])

    #steps = torch.Tensor([40, 40, 40, 40])
    #lowers = torch.Tensor([
    #    0,
    #    -A.PI, 
    #    -A.MAX_VEL_1, 
    #    -A.MAX_VEL_2])
    #uppers = torch.Tensor([
    #    2*A.PI,
    #    A.PI, 
    #    A.MAX_VEL_1, 
    #    A.MAX_VEL_2])
    #vi = AnalyticValueIter("reg", A, .001, 
    #               lowers, uppers, steps, midpoints, eps=.1,
    #               use_cuda=use_cuda, use_geom=True, periodic=[0,1], R=0)
    #return vi

    geom = torch.Tensor([1.14, 1.14, 1.1, 1.1])
    steps = torch.Tensor([100, 100, 40, 40])
    lowers = torch.Tensor([
        0,
        -A.PI, 
        -A.MAX_VEL_1, 
        -A.MAX_VEL_2])
    uppers = torch.Tensor([
        2*A.PI,
        A.PI, 
        A.MAX_VEL_1, 
        A.MAX_VEL_2])
    vi_geom7 = AnalyticValueIter("geom7", A, .001, 
                   lowers, uppers, steps, midpoints, eps=.00001,
                   use_cuda=use_cuda, geom=geom, periodic=[0,1], R=0)
    return vi_geom7

    geom = torch.Tensor([1.14, 1.14, 1.1, 1.1])
    steps = torch.Tensor([100, 100, 40, 40])
    lowers = torch.Tensor([
        0,
        -A.PI, 
        -A.MAX_VEL_1, 
        -A.MAX_VEL_2])
    uppers = torch.Tensor([
        2*A.PI,
        A.PI, 
        A.MAX_VEL_1, 
        A.MAX_VEL_2])
    vi_geom6 = AnalyticValueIter("geom6", A, .0005, 
                   lowers, uppers, steps, midpoints, eps=.00001,
                   use_cuda=use_cuda, geom=geom, periodic=[0,1], R=0)
    return vi_geom6

    geom = torch.Tensor([1.5, 1.5, 1.1, 1.1])
    steps = torch.Tensor([40, 40, 40, 40])
    lowers = torch.Tensor([
        0,
        -A.PI, 
        -A.MAX_VEL_1, 
        -A.MAX_VEL_2])
    uppers = torch.Tensor([
        2*A.PI,
        A.PI, 
        A.MAX_VEL_1, 
        A.MAX_VEL_2])
    vi_geom4 = AnalyticValueIter("geom4", A, .0005, #.001 for first 40k steps
                   lowers, uppers, steps, midpoints, eps=.00001,
                   use_cuda=use_cuda, geom=geom, periodic=[0,1], R=0)
    return vi_geom4

    # coarse
    steps = torch.Tensor([40, 40, 40, 40])
    lowers = torch.Tensor([
        0,
        -A.PI, 
        -A.MAX_VEL_1, 
        -A.MAX_VEL_2])
    uppers = torch.Tensor([
        2*A.PI,
        A.PI, 
        A.MAX_VEL_1, 
        A.MAX_VEL_2])
    vi_coarse = AnalyticValueIter("coarse_analytic", A, .0001, 
                   lowers, uppers, steps, midpoints, .01,
                   use_cuda=use_cuda, periodic=[0,1], R=0)
    #return vi_coarse

    #steps = torch.Tensor([16, 16, 80, 80])
    #lowers = torch.Tensor([
    #    0,
    #    -A.PI, 
    #    -A.MAX_VEL_1, 
    #    -A.MAX_VEL_2])
    #uppers = torch.Tensor([
    #    2*A.PI,
    #    A.PI, 
    #    A.MAX_VEL_1, 
    #    A.MAX_VEL_2])
    ##vi_coarse = AnalyticValueIter("save", A, .0001, 
    ##               lowers, uppers, steps, midpoints, .01,
    ##               use_cuda=use_cuda, periodic=[0,1], R=0)
    #vi_coarse = AnalyticValueIter(A, .0001, 
    #               lowers, uppers, steps, midpoints,
    #               use_cuda=use_cuda, periodic=[0,1], R=0)
    #return vi_coarse

    # semi-coarse
    steps = torch.Tensor([40, 40, 40, 40])
    lowers = torch.Tensor([
        A.PI - A.PI/2, 
        -A.PI/2, 
        -A.MAX_VEL_1/2, 
        -A.MAX_VEL_2/2])
    uppers = torch.Tensor([
        A.PI + A.PI/2,
        A.PI/2, 
        A.MAX_VEL_1/2, 
        A.MAX_VEL_2/2])
    vi_semi_coarse = AnalyticValueIter("semi_coarse", A, .0001, 
                   lowers, uppers, steps, midpoints, .4,
                   use_cuda=use_cuda, periodic=[], R=0,
                   parent=[vi_coarse]) #R=.005)
    #return vi_semi_coarse

    # med-coarse
    steps = torch.Tensor([40, 40, 40, 40])
    lowers = torch.Tensor([
        A.PI - A.PI/4, 
        -A.PI/4, 
        -A.MAX_VEL_1/4, 
        -A.MAX_VEL_2/4])
    uppers = torch.Tensor([
        A.PI + A.PI/4,
        A.PI/4, 
        A.MAX_VEL_1/4, 
        A.MAX_VEL_2/4])
    vi_med_coarse = AnalyticValueIter("med_coarse", A, .0001, 
                   lowers, uppers, steps, midpoints, .2,
                   use_cuda=use_cuda, periodic=[], R=0,
                   parent=[vi_semi_coarse, vi_coarse])
    #return vi_med_coarse

    # fine
    steps = torch.Tensor([40, 40, 40, 40])
    lowers = torch.Tensor([
        A.PI - A.PI/8, 
        -A.PI/8, 
        -A.MAX_VEL_1/8, 
        -A.MAX_VEL_2/8])
    uppers = torch.Tensor([
        A.PI + A.PI/8,
        A.PI/8, 
        A.MAX_VEL_1/8, 
        A.MAX_VEL_2/8])
    vi_fine = AnalyticValueIter("fine_analytic", A, .00001, 
                   lowers, uppers, steps, midpoints, .1,
                   use_cuda=use_cuda, periodic=[], R=0,
                   parent=[vi_med_coarse, vi_semi_coarse, vi_coarse])
    #return vi_fine

    # semifine
    steps = torch.Tensor([40, 40, 40, 40])
    lowers = torch.Tensor([
        A.PI - A.PI/24, 
        -A.PI/24, 
        -A.MAX_VEL_1/24, 
        -A.MAX_VEL_2/24])
    uppers = torch.Tensor([
        A.PI + A.PI/24,
        A.PI/24, 
        A.MAX_VEL_1/24, 
        A.MAX_VEL_2/24])
    vi_semifine = AnalyticValueIter("semifine", A, .00001, 
                   lowers, uppers, steps, midpoints, .01,
                   use_cuda=use_cuda, periodic=[], R=0,
                   parent=[vi_fine, vi_med_coarse, vi_semi_coarse, vi_coarse])
    #return vi_semifine

    # superfine
    steps = torch.Tensor([4, 4, 320, 320])
    lowers = torch.Tensor([
        A.PI - A.PI/80, 
        -A.PI/80, 
        -A.MAX_VEL_1/4, 
        -A.MAX_VEL_2/4])
    uppers = torch.Tensor([
        A.PI + A.PI/80,
        A.PI/80, 
        A.MAX_VEL_1/4, 
        A.MAX_VEL_2/4])
    vi = AnalyticValueIter("superfine3", A, .0001, 
                   lowers, uppers, steps, midpoints, .000001,
                   use_cuda=use_cuda, periodic=[], R=0,
                   parent=[vi_semifine, vi_fine, 
                           vi_med_coarse, vi_semi_coarse, 
                           vi_coarse])
    return vi

def make_pendulum_vi(use_cuda=True, theta_steps=320, dtheta_steps=320):
    P = PendulumNet
    lowers = torch.Tensor([
        -P.PI, 
        -P.MAX_VEL, 
        -P.TORQUE_LIMIT])
    uppers = torch.Tensor([
        P.PI, 
        P.MAX_VEL, 
        P.TORQUE_LIMIT])
    midpoints = torch.Tensor([
        0,
        0,
        0])
    steps = torch.Tensor([theta_steps, dtheta_steps, 2])
    vi = ValueIter("pend", P, .0005, 
                   lowers, uppers, steps, midpoints, 
                   use_cuda=use_cuda)
    return vi

def make_analytic_pendulum_vi(use_cuda=True, theta_steps=320, dtheta_steps=320):
    P = AnalyticPendulumNet
    lowers = torch.Tensor([
        -P.PI, 
        -P.MAX_VEL])
    uppers = torch.Tensor([
        P.PI, 
        P.MAX_VEL])
    midpoints = torch.Tensor([
        0,
        0])
    #geom = torch.Tensor([1.5, 1.3])
    geom = torch.Tensor([1.3, 1.1])
    #geom = None
    steps = torch.Tensor([theta_steps, dtheta_steps])
    vi = AnalyticValueIter("pend3", P, .005, 
                           lowers, uppers, steps, midpoints, eps=.0001,
                           use_cuda=use_cuda, geom=geom)
    return vi

if __name__ == "__main__":
    #results = []
    #for theta in range(20, 340, 20):
    #    for dtheta in range(20, 340, 20):
    #        vi = make_pendulum_vi(theta, dtheta)
    #        name = f"{theta}_{dtheta}"
    #        it, err = vi.run(eps=.0001, err_tol=.0005, name=name, max_iter=100000)
    #        result = {"theta": theta,
    #                  "dtheta": dtheta,
    #                  "it": it,
    #                  "err": err.item()}
    #        print(result)
    #        results.append(result)
    #        with open("results.pickle", "wb") as f:
    #            pickle.dump(results, f)

    #theta = 20
    #dtheta = 20
    #vi = make_pendulum_vi(theta, dtheta)
    #name = f"{theta}_{dtheta}"
    #it, err = vi.run(eps=.0001, err_tol=.0005, name=name, max_iter=100000)
    #vi.simulate(name=name)

    #theta = 320
    #vi = make_pendulum_vi(theta, dtheta)
    #name = f"{theta}_{dtheta}"
    #it, err = vi.run(eps=.0001, err_tol=.0005, name=name, max_iter=100000)
    #vi.simulate(name=name)

    #vi = make_acrobot_vi()
    #vi.run()
    #vi.simulate()


    #vi = make_analytic_acrobot_vi()
    ###J = np.load(f"outputs/precompute/bot_analytic_ctg.npy")
    ###a = np.load(f"outputs/precompute/bot_analytic_a.npy")
    #vi.simulate()
    #vi.run()
    #vi.simulate()

    #vi = make_pendulum_vi(theta_steps=320, dtheta_steps=320)
    #it, _ = vi.run(err_tol=.0005)
    #print(it)
    #vi.simulate()

    vi = make_analytic_pendulum_vi(theta_steps=80, dtheta_steps=80)
    #vi = make_analytic_pendulum_vi()
    #vi = make_pendulum_vi()
    it, _ = vi.run(err_tol=.0005)
    print(it)
    vi.simulate()

