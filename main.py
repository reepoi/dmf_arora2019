import json
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import cvxpy as cvx

DMF_VECTOR_FIELDS_DATA_DIR = Path(__file__).parent / '..' / '..' / 'dmf_vector_fields' / 'data'
from dmf_vector_fields import data, model as vf_model, enums

import lunzi as lz
from lunzi.typing import *
from opt import GroupRMSprop

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
rng = np.random.RandomState(seed=20210909)


class FLAGS(lz.BaseFLAGS):
    problem = ''
    gt_path = ''
    obs_path = ''

    depth = 1
    n_train_samples = 0
    n_iters = 1000000
    n_dev_iters = max(1, n_iters // 1000)
    init_scale = 0.001  # average magnitude of entries
    shape = [0, 0]
    n_singulars_save = 0

    optimizer = 'GroupRMSprop'
    initialization = 'gaussian'  # `orthogonal` or `identity` or `gaussian`
    lr = 0.01
    train_thres = 1e-7

    hidden_sizes = []

    technique = 'identity'
    mask_rate = 0.9
    grid_density = 100

    @classmethod
    def finalize(cls):
        assert cls.problem
        cls.technique = enums.Technique(cls.technique)
        if cls.problem in {e.value for e in enums.DataSet}:
            cls.problem = enums.DataSet(cls.problem)
            cls.shape = ProblemBroker(cls.problem, cls.technique, cls.mask_rate, cls.depth, cls.grid_density).problem_shape()
        cls.add('hidden_sizes', [cls.shape[0]] + [cls.shape[1]] * cls.depth, overwrite_false=True)


def get_e2e(model):
    weight = None
    for fc in model.children():
        assert isinstance(fc, nn.Linear) and fc.bias is None
        if weight is None:
            weight = fc.weight.t()
        else:
            weight = fc(weight)

    return weight


@FLAGS.inject
def init_model(model, *, hidden_sizes, initialization, init_scale, _log):
    depth = len(hidden_sizes) - 1

    if initialization == 'orthogonal':
        scale = (init_scale * np.sqrt(hidden_sizes[0]))**(1. / depth)
        matrices = []
        for param in model.parameters():
            nn.init.orthogonal_(param)
            param.data.mul_(scale)
            matrices.append(param.data.cpu().numpy())
        for a, b in zip(matrices, matrices[1:]):
            assert np.allclose(a.dot(a.T), b.T.dot(b), atol=1e-6)
    elif initialization == 'identity':
        scale = init_scale**(1. / depth)
        for param in model.parameters():
            nn.init.eye_(param)
            param.data.mul_(scale)
    elif initialization == 'gaussian':
        n = hidden_sizes[0]
        # assert hidden_sizes[0] == hidden_sizes[-1]
        scale = init_scale**(1. / depth) * n**(-0.5)
        for param in model.parameters():
            nn.init.normal_(param, std=scale)
        e2e = get_e2e(model).detach().cpu().numpy()
        e2e_fro = np.linalg.norm(e2e, 'fro')
        desired_fro = FLAGS.init_scale * np.sqrt(n)
        _log.info(f"[check] e2e fro norm: {e2e_fro:.6e}, desired = {desired_fro:.6e}")
        # assert 0.8 <= e2e_fro / desired_fro <= 1.2
    elif initialization == 'uniform':
        n = hidden_sizes[0]
        assert hidden_sizes[0] == hidden_sizes[-1]
        scale = np.sqrt(3.) * init_scale**(1. / depth) * n**(-0.5)
        for param in model.parameters():
            nn.init.uniform_(param, a=-scale, b=scale)
        e2e = get_e2e(model).detach().cpu().numpy()
        e2e_fro = np.linalg.norm(e2e, 'fro')
        desired_fro = FLAGS.init_scale * np.sqrt(n)
        _log.info(f"[check] e2e fro norm: {e2e_fro:.6e}, desired = {desired_fro:.6e}")
        assert 0.8 <= e2e_fro / desired_fro <= 1.2
    else:
        assert 0


class BaseProblem:
    def __init__(self, name):
        self.name = name

    def get_d_e2e(self, e2e):
        pass

    def get_train_loss(self, e2e):
        pass

    def get_test_loss(self, e2e):
        pass

    def get_cvx_opt_constraints(self, x) -> list:
        pass


@FLAGS.inject
def cvx_opt(prob: BaseProblem, *, shape, _log: Logger, _writer: SummaryWriter, _fs: FileStorage):
    x = cvx.Variable(shape=shape)

    objective = cvx.Minimize(cvx.norm(x, 'nuc'))
    constraints = prob.get_cvx_opt_constraints(x)

    problem = cvx.Problem(objective, constraints)
    problem.solve(solver=cvx.SCS, verbose=True, use_indirect=False)
    e2e = torch.from_numpy(x.value).float()

    train_loss = prob.get_train_loss(e2e)
    test_loss = prob.get_test_loss(e2e)

    nuc_norm = e2e.norm('nuc')
    _log.info(f"train loss = {train_loss.item():.3e}, "
              f"test error = {test_loss.item():.3e}, "
              f"nuc_norm = {nuc_norm.item():.3f}")
    _writer.add_scalar('loss/train', train_loss.item())
    _writer.add_scalar('loss/test', test_loss.item())
    _writer.add_scalar('nuc_norm', nuc_norm.item())

    torch.save(e2e, _fs.resolve('$LOGDIR/nuclear.npy'))


class MatrixCompletion(BaseProblem):
    ys: torch.Tensor  # these are the sampled data of the ground truth matrix

    @FLAGS.inject
    def __init__(self, *, gt_path, obs_path):
        self.matrix = torch.load(gt_path, map_location=device)
        (self.us, self.vs), self.ys_ = torch.load(obs_path, map_location=device)
        super().__init__('')

    def get_train_loss(self, e2e):
        self.ys = e2e[self.us, self.vs]
        return (self.ys - self.ys_).pow(2).mean()

    def get_test_loss(self, e2e):
        return vf_model.norm_mean_abs_error(e2e, self.matrix)

    @FLAGS.inject
    def get_d_e2e(self, e2e, shape):
        d_e2e = torch.zeros(shape, device=device)
        d_e2e[self.us, self.vs] = self.ys - self.ys_
        d_e2e = d_e2e / len(self.ys_)
        return d_e2e

    @FLAGS.inject
    def get_cvx_opt_constraints(self, x, shape):
        A = np.zeros(shape)
        mask = np.zeros(shape)
        A[self.us, self.vs] = self.ys_
        mask[self.us, self.vs] = 1
        eps = 1.e-3
        constraints = [cvx.abs(cvx.multiply(x - A, mask)) <= eps]
        return constraints


class MyMatrixCompletion(BaseProblem):
    def __init__(self, name, matrix, mask):
        self.matrix = matrix
        self.mask = mask
        super().__init__(name)

    def get_d_e2e(self, e2e):
        pass

    def get_train_loss(self, e2e):
        return vf_model.mean_sqrd_error_masked(self.matrix, e2e, self.mask)

    def get_test_loss(self, e2e):
        return vf_model.norm_mean_abs_error(e2e, self.matrix)


class ProblemBroker:
    def __init__(self, problem, technique, mask_rate, num_factors, grid_density):
        mask = None
        if problem is enums.DataSet.DOUBLE_GYRE:
            self.vbt = data.double_gyre()
        elif problem is enums.DataSet.ANEURYSM:
            self.vbt = data.VelocityByTimeAneurysm.from_save(DMF_VECTOR_FIELDS_DATA_DIR / 'aneurysm' / 'vel_by_time')
        elif problem is enums.DataSet.FUNC1:
            self.vbt = data.func1()
        elif problem is enums.DataSet.ARORA2019_5:
            fp = DMF_VECTOR_FIELDS_DATA_DIR / 'arora2019' / 'rank5.pt'
            tf = data.MatrixArora2019(time=0, filepath=fp)
            self.vbt = data.VelocityByTime.from_vec_fields([tf.vec_field])
            mask = tf.saved_mask('0.8')
        elif problem is enums.DataSet.CAMERAMAN:
            fp = DMF_VECTOR_FIELDS_DATA_DIR / 'images' / 'cameraman.jpg'
            tf = data.MatrixImage(time=0, filepath=fp)
            self.vbt = data.VelocityByTime.from_vec_fields([tf.vec_field])
        elif problem is enums.DataSet.CHECKERBOARD:
            fp = DMF_VECTOR_FIELDS_DATA_DIR / 'images' / 'checkerboard.png'
            tf = data.MatrixImage(time=0, filepath=fp)
            self.vbt = data.VelocityByTime.from_vec_fields([tf.vec_field])
        elif problem is enums.DataSet.SKELETON:
            fp = DMF_VECTOR_FIELDS_DATA_DIR / 'images' / 'skeleton.png'
            tf = data.MatrixImage(time=0, filepath=fp)
            self.vbt = data.VelocityByTime.from_vec_fields([tf.vec_field])
        elif problem is enums.DataSet.FOX:
            fp = DMF_VECTOR_FIELDS_DATA_DIR / 'images' / 'fox.jpg'
            tf = data.MatrixImage(time=0, filepath=fp)
            self.vbt = data.VelocityByTime.from_vec_fields([tf.vec_field])
        self.technique = technique
        self.grid_density = grid_density
        self.mask_rate = mask_rate
        self.num_factors = num_factors
        self.mask = mask
        if self.mask is None:
            self.mask = vf_model.get_bit_mask(self.problem_shape(), self.mask_rate)

    def problem_shape(self):
        if self.technique is not enums.Technique.INTERPOLATED:
            return self.vbt.shape_as_completable(interleaved=self.technique is enums.Technique.INTERPOLATED)
        return (self.grid_density,) * 2

    def layers(self):
        rows, cols = self.problem_shape()
        min_dim = min(rows, cols)
        matrix_factor_dimensions = [(rows, min_dim)]
        matrix_factor_dimensions += [(min_dim, min_dim) for _ in range(1, self.num_factors - 1)]
        matrix_factor_dimensions.append((min_dim, cols))
        return matrix_factor_dimensions

    def problems(self):
        mask = torch.tensor(self.mask).to(device)
        vbt = self.vbt.numpy_to_torch().as_completable(interleaved=self.technique is enums.Technique.INTERLEAVED)
        if self.technique is not enums.Technique.INTERPOLATED:
            return tuple(MyMatrixCompletion(n, a, mask) for n, a in zip(vbt.components, vbt.vel_by_time_axes))
        problems = []
        for t in range(self.vbt.timeframes):
            vf = self.vbt.timeframe(t).as_completable(self.grid_density).numpy_to_torch().vec_field
            for n, a in zip(vf.components, vf.vel_axes):
                problems.append(MyMatrixCompletion(f'{n} (t = {t})', a, mask))
        return tuple(problems)

    def reconstruct_vbt(self, results):
        coords_torch = self.vbt.timeframe(0).numpy_to_torch().vec_field.coords
        if self.technique is enums.Technique.IDENTITY:
            return self.vbt.__class__(
                filepath=None,
                coords=coords_torch,
                vel_by_time_axes=results,
                components=data.auto_component_names(len(results))
            ).torch_to_numpy()
        elif self.technique is enums.Technique.INTERLEAVED:
            dims = len(self.vbt.components)
            return self.vbt.__class__(
                filepath=None,
                coords=coords_torch,
                vel_by_time_axes=tuple(results[0][i::dims] for i in range(dims)),
                components=data.auto_component_names(dims)
            ).torch_to_numpy()
        elif self.technique is enums.Technique.INTERPOLATED:
            coords_interp = self.vbt.coords.bounding_grid(self.grid_density)
            idx = 0
            vec_fields = []
            for _ in range(self.vbt.timeframes):
                vel_axes = []
                for _ in self.vbt.components:
                    vel_axes.append(results[idx].detach().cpu().numpy())
                    idx += 1
                vec_fields.append(data.VectorField(coords=coords_interp, vel_axes=vel_axes))
            vfs_interp = tuple(vf.interp(coords=self.vbt.coords) for vf in vec_fields)
            return self.vbt.__class__.from_vec_fields(vfs_interp)


class MatrixSensing(BaseProblem):
    ys: torch.Tensor

    @FLAGS.inject
    def __init__(self, *, gt_path, obs_path):
        self.w_gt = torch.load(gt_path, map_location=device)
        self.xs, self.ys_ = torch.load(obs_path, map_location=device)
        super().__init__('')

    def get_train_loss(self, e2e):
        self.ys = (self.xs * e2e).sum(dim=-1).sum(dim=-1)
        return (self.ys - self.ys_).pow(2).mean()

    def get_test_loss(self, e2e):
        return (self.w_gt - e2e).view(-1).pow(2).mean()

    @FLAGS.inject
    def get_d_e2e(self, e2e, shape):
        d_e2e = self.xs.view(-1, *shape) * (self.ys - self.ys_).view(len(self.xs), 1, 1)
        d_e2e = d_e2e.sum(0)
        return d_e2e

    def get_cvx_opt_constraints(self, X):
        eps = 1.e-3
        constraints = []
        for x, y_ in zip(self.xs, self.ys_):
            constraints.append(cvx.abs(cvx.sum(cvx.multiply(X, x)) - y_) <= eps)
        return constraints


class MovieLens100k(BaseProblem):
    ys: torch.Tensor

    @FLAGS.inject
    def __init__(self, *, obs_path, n_train_samples):
        (self.us, self.vs), ys_ = torch.load(obs_path, map_location=device)
        # self.ys_ = (ys_ - ys_.mean()) / ys_.std()
        self.ys_ = ys_
        self.n_train_samples = n_train_samples
        super().__init__('')

    def get_train_loss(self, e2e):
        self.ys = e2e[self.us[:self.n_train_samples], self.vs[:self.n_train_samples]]
        return (self.ys - self.ys_[:self.n_train_samples]).pow(2).mean()

    def get_test_loss(self, e2e):
        ys = e2e[self.us[self.n_train_samples:], self.vs[self.n_train_samples:]]
        return (ys - self.ys_[self.n_train_samples:]).pow(2).mean()

    @FLAGS.inject
    def get_d_e2e(self, e2e, *, shape):
        d_e2e = torch.zeros(shape, device=device)
        d_e2e[self.us[:self.n_train_samples], self.vs[:self.n_train_samples]] = \
            self.ys - self.ys_[:self.n_train_samples]
        d_e2e = d_e2e / len(self.ys_)
        return d_e2e

    @FLAGS.inject
    def get_cvx_opt_constraints(self, x, *, shape):
        A = np.zeros(shape)
        mask = np.zeros(shape)
        A[self.us[:self.n_train_samples], self.vs[:self.n_train_samples]] = self.ys_[:self.n_train_samples]
        mask[self.us[:self.n_train_samples], self.vs[:self.n_train_samples]] = 1
        eps = 1.e-3
        constraints = [cvx.abs(cvx.multiply(x - A, mask)) <= eps]
        return constraints


@lz.main(FLAGS)
@FLAGS.inject
def main(*, depth, hidden_sizes, mask_rate, technique, grid_density, n_iters, problem, train_thres, _seed, _log, _writer, _info, _fs):
    prob: BaseProblem
    if problem == 'matrix-completion':
        prob = MatrixCompletion()
    elif problem == 'matrix-sensing':
        prob = MatrixSensing()
    elif problem == 'ml-100k':
        prob = MovieLens100k()
    elif problem in enums.DataSet:
        prob = ProblemBroker(problem, technique, mask_rate, depth, grid_density)

    layers = zip(hidden_sizes, hidden_sizes[1:])
    model = nn.Sequential(*[nn.Linear(f_in, f_out, bias=False) for (f_in, f_out) in layers]).to(device)
    _log.info(model)

    if FLAGS.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), FLAGS.lr)
    elif FLAGS.optimizer == 'GroupRMSprop':
        optimizer = GroupRMSprop(model.parameters(), FLAGS.lr, eps=1e-4)
    elif FLAGS.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), FLAGS.lr)
    elif FLAGS.optimizer == 'cvxpy':
        cvx_opt(prob)
        return
    else:
        raise ValueError

    init_model(model)

    problems = (prob,)
    if isinstance(prob, ProblemBroker):
        problems = prob.problems()

    results = []
    for p in problems:
        _log.info(f'***** BEGIN {problem}: {p.name} *****')

        loss = None
        for T in range(n_iters):
            e2e = get_e2e(model)

            loss = p.get_train_loss(e2e)

            params_norm = 0
            for param in model.parameters():
                params_norm = params_norm + param.pow(2).sum()
            optimizer.zero_grad()
            loss.backward()

            with torch.no_grad():
                test_loss = p.get_test_loss(e2e)

                if T % FLAGS.n_dev_iters == 0 or loss.item() <= train_thres:

                    U, singular_values, V = e2e.svd()  # U D V^T = e2e
                    schatten_norm = singular_values.pow(2. / depth).sum()

                    grads = [param.grad.cpu().data.numpy().reshape(-1) for param in model.parameters()]
                    grads = np.concatenate(grads)
                    avg_grads_norm = np.sqrt(np.mean(grads**2))
                    avg_param_norm = np.sqrt(params_norm.item() / len(grads))

                    if isinstance(optimizer, GroupRMSprop):
                        adjusted_lr = optimizer.param_groups[0]['adjusted_lr']
                    else:
                        adjusted_lr = optimizer.param_groups[0]['lr']
                    _log.info(f"Iter #{T}: train = {loss.item():.3e}, test = {test_loss.item():.3e}, "
                              f"Schatten norm = {schatten_norm:.3e}, "
                              f"grad: {avg_grads_norm:.3e}, "
                              f"lr = {adjusted_lr:.3f}")

                    _writer.add_scalar('loss/train', loss.item(), global_step=T)
                    _writer.add_scalar('loss/test', test_loss, global_step=T)
                    _writer.add_scalar('Schatten_norm', schatten_norm, global_step=T)
                    _writer.add_scalar('norm/grads', avg_grads_norm, global_step=T)
                    _writer.add_scalar('norm/params', avg_param_norm, global_step=T)

                    for i in range(FLAGS.n_singulars_save):
                        _writer.add_scalar(f'singular_values/{i}', singular_values[i], global_step=T)

                    torch.save(e2e, _fs.resolve("$LOGDIR/final.npy"))
                    if loss.item() <= train_thres:
                        break
            optimizer.step()

        _log.info(f"train loss = {loss.item()}. test loss = {test_loss.item()}")
        _log.info(f'***** END {problem}: {p.name} *****')
        results.append(e2e)

    if isinstance(prob, ProblemBroker):
        vbt = prob.vbt
        vbt_rec = prob.reconstruct_vbt(results)
        nmaes = {n: vf_model.norm_mean_abs_error(rec_a, a, lib=np) for n, rec_a, a in zip(vbt.components, vbt_rec.vel_by_time_axes, vbt.vel_by_time_axes)}
        vbt_rec.save(lz.fs.log_dir / 'reconstructed', plot_time=0)

        _log.info(f'FINAL NMAE: {json.dumps(nmaes)}')


if __name__ == '__main__':
    main()
