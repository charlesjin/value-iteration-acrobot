import numpy as np

class UniformMesh(object):
    def __init__(self, lowers, uppers, steps, data_dims=2, data_init=None, points_init=None):
        self.uppers = uppers
        self.lowers = lowers
        self.steps = steps
        self.step_sizes = (uppers - lowers) / steps
        self.data_dims = data_dims

        if data_init is not None:
            self.data = data_init
        else:
            #data_dims = np.append(self.data_dims, self.steps+1)
            self.data = [np.zeros(self.steps+1,dtype=np.float16) for _ in range(self.data_dims)]

        N = len(self.uppers)
        self.corners = ((np.arange(2**N)[:,None] & (1 << np.arange(N))) > 0).astype(int)

        if points_init is not None:
            self.points = points_init
        else:
            grid = [np.linspace(l,u,s+1) for l,u,s in zip(lowers,uppers,steps)]
            grid = np.meshgrid(*grid)
            grid = [ax.ravel() for ax in grid]
            # in global coordinates
            self.points = np.c_[tuple(grid)]

        self.num_points = len(self.points)

    def get_grid_coords(self, points):
        sides = self.uppers - self.lowers
        assert np.all(points >= self.lowers) and np.all(self.uppers)
        local_coords = (points - self.lowers) * self.steps / sides

        global_coords = np.floor((points - self.lowers) * self.steps / sides).astype(int)
        global_coords[global_coords == self.steps] -= 1
        local_coords -= global_coords
        return global_coords, local_coords

    def set_data_from_flat(self, data):
        assert data_dim < self.data_dims
        self.data[data_dim] = data.reshape(self.steps+1)

    def interpolate(self, points, data_dim=0, dot=True):
        N = self.corners.shape[-1]

        global_coords, local_coords = self.get_grid_coords(points)

        ###
        #corners = global_coords.reshape(-1, N)
        #idxs = tuple([corners[:,i] for i in range(N)])
        #return idxs #self.data[data_dim][idxs]
        ###

        global_coords = np.expand_dims(global_coords, axis=1)
        # shape: num_points x 2^mesh_dim x mesh_dim
        corners = global_coords + self.corners

        # shape: num_points * 2^mesh_dim x mesh_dim
        corners = corners.reshape(-1, N)

        idxs = tuple([corners[:,i] for i in range(N)])
        data = self.data[data_dim][idxs]
        # shape: num_points * 2^mesh_dim
        data = data.reshape(len(points), -1)

        local_weights = np.stack([self.corners] * len(points), axis=0)
        local_weights = np.transpose(local_weights, (0, 2, 1))
        local_coords = np.expand_dims(local_coords, axis=-1)
        local_weights = local_weights * local_coords + (1 - local_weights) * (1 - local_coords)
        # shape: num_points * 2^mesh_dim
        local_weights = np.prod(local_weights, axis=1)
        # sum along last dim = 1
        #print(np.sum(local_weights, axis=-1))

        if dot:
            return np.sum(local_weights * data, axis=1)
        else:
            return local_weights, data

#class AdaptiveUniformMesh(UniformMesh):
#    def __init__(self, lowers, uppers, steps, data_dims=2, data_init=None):
#        super().__init__(lowers, uppers, steps, data_dims=data_dims, data_init=data_init)
#        self.submeshes = np.zeros(self.steps, dtype=object)
#
#    # TODO interpolate in the submesh?
#    def interpolate(self, points, data_dim=0):
#        N = self.corners.shape[-1]
#
#        global_coords, local_coords = self.get_grid_coords(points)
#        global_coords = np.expand_dims(global_coords, axis=1)
#        # shape: num_points x 2^mesh_dim x mesh_dim
#        corners = global_coords + self.corners
#
#        # shape: num_points * 2^mesh_dim x mesh_dim
#        corners = corners.reshape(-1, N)
#
#        idxs = tuple([corners[:,i] for i in range(N)])
#        data = self.data[data_dim][idxs]
#        # shape: num_points * 2^mesh_dim
#        data = data.reshape(len(points), -1)
#
#        local_weights = np.stack([self.corners] * len(points), axis=0)
#        local_weights = np.transpose(local_weights, (0, 2, 1))
#        local_coords = np.expand_dims(local_coords, axis=-1)
#        local_weights = local_weights * local_coords + (1 - local_weights) * (1 - local_coords)
#        # shape: num_points * 2^mesh_dim
#        # sum along last dim = 1
#        local_weights = np.prod(local_weights, axis=1)
#        
#        return np.sum(local_weights * data, axis=1)
#
#    def refine(self, points, steps, data_dims=0):
#        if not isinstance(data_dims, list):
#            data_dims = [data_dims]
#
#        coords, _ = self.get_grid_coords(points)
#        sides = self.uppers - self.lowers
#        lowers = coords     * sides / self.steps + self.lowers
#        uppers = (1+coords) * sides / self.steps + self.lowers
#
#        new = []
#        new_idxs = []
#        propogate = {}
#        for point, coord, lower, upper, step in zip(points, coords, lowers, uppers, steps):
#            idx = tuple(coord)
#            if idx in new_idxs:
#                print("skipping " + str(idx))
#            elif self.submeshes[idx] == 0:
#                print("refining " + str(idx) + "...")
#                submesh = AdaptiveUniformMesh(lower, upper, step, data_dims=self.data_dims)
#                for dim in data_dims:
#                    submesh.set_data_from_flat(
#                            self.interpolate(submesh.points, data_dim=dim), dim)
#                self.submeshes[idx] = submesh
#                new.append(submesh)
#                new_idxs.append(idx)
#            else:
#                to_prop = propogate.get(idx, [])
#                to_prop.append((point, step))
#                propogate[idx] = to_prop
#        for idx, vals in propogate.items():
#            print("propogating " + str(idx) + "...")
#            submeshes = self.submeshes[idx].refine(*zip(*vals))
#            new.extend(submeshes)
#        return new

if __name__ == "__main__":
    #lower = np.array([0,0,0,0])
    #upper = np.array([10,10,10,10])
    #steps = np.array([2,2,2,2])
    #base = AdaptiveUniformMesh(lower, upper, steps)

    #points = np.array([[0,0,0,0],[1.1,3.1,3.1,4.1], [8,8,3,8], [10,10,10,10]])
    #print(base.get_grid_coords(points))

    #print()
    #points = np.array([[1,3,3,4], [9.5,9.5,9.5,9.5], [10,10,10,10]])
    #steps = np.array([[2,2,2,2],[2,2,2,2],[2,2,2,2]])
    #base.refine(points, steps)

    #print()
    #points = np.array([[1,3,3,4], [9,9.5,9.5,9.5], [9.5,10,10,10]])
    #steps = np.array([[2,2,2,2],[2,2,2,2],[2,2,2,2]])
    #base.refine(points, steps)

    #print()
    #print(base.interpolate(points))

    lower = np.array([0, 0])
    upper = np.array([10, 10])
    steps = np.array([10, 10])
    data = np.array([np.arange(0, 11, 1)] * 11)
    base = UniformMesh(lower, upper, steps, data_init = [data])

    points = np.array([[10, 10], [5.5, 5.5], [5.75, 5.5], [5.5, 5.75]])
    v = base.interpolate(points)
    assert v[0] == 10
    assert v[1] == 5.5
    assert v[2] == 5.5
    assert v[3] == 5.75

    for i in range(11):
        for j in range(11):
            data[i,j] = i*j
    v = base.interpolate(points)
    assert v[0] == 100
    assert v[1] == 25 / 4 + 30 / 4 + 30 / 4 + 36 / 4
    assert v[2] == 25 / 8 + 30 * 3 / 8 + 30 / 8 + 36 * 3 / 8
    assert v[3] == 25 / 8 + 30 * 3 / 8 + 30 / 8 + 36 * 3 / 8
    assert v[2] > v[1]
    assert v[2] == v[3]

