from tqdm import tqdm
import torch
import numpy as np

from continuous.acrobot_net import AcrobotNet
from continuous.pendulum_net import PendulumNet
from continuous.net import ValueIter

from continuous.analytic_acrobot_net import AnalyticAcrobotNet
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
    midpoints = torch.Tensor([
        A.PI,
        0,
        0,
        0])
    steps = torch.Tensor([40, 40, 40, 40])
    vi = AnalyticValueIter(A, .01, 
                   lowers, uppers, steps, midpoints, 
                   use_cuda=use_cuda, periodic=[0,1], R=0) #R=.005)
    return vi

def make_pendulum_vi(use_cuda=True):
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
    steps = torch.Tensor([20, 20, 2])
    vi = ValueIter(P, .01, 
                   lowers, uppers, steps, midpoints, 
                   use_cuda=use_cuda)
    return vi

if __name__ == "__main__":
    #vi = make_pendulum_vi()
    #vi.run()
    #vi.simulate()

    #vi = make_acrobot_vi()
    #vi.run()
    #vi.simulate()

    #J = np.load(f"outputs/precompute/top_analytic_ctg.npy")
    vi = make_analytic_acrobot_vi()
    vi.run()
    vi.simulate()

