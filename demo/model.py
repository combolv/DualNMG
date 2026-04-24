import taichi as ti
import numpy as np
import math
import time
from itertools import product as prod

@ti.data_oriented
class Model:
    def __init__(self, res, base_level=3, real=float, hidden_dim=8):
        # parameters
        self.res = res
        self.n_mg_levels = int(math.log2(min(res))) - base_level + 1
        self.real = real

        self.pre_and_post_smoothing = 2
        self.bottom_smoothing = 10

        # rhs
        self.b = ti.field(dtype=real, shape=res)  # Ax=b

        self.r = [
            ti.field(
                dtype=ti.f32, shape=[res[0] // 2**l, res[1] // 2**l, res[2] // 2**l]
            )
            for l in range(self.n_mg_levels)
        ]  # residual
        self.z = [
            ti.field(
                dtype=ti.f32, shape=[res[0] // 2**l, res[1] // 2**l, res[2] // 2**l]
            )
            for l in range(self.n_mg_levels)
        ]  # M^-1 self.r

        # lhs
        self.is_dof = [
            ti.field(
                dtype=bool, shape=[res[0] // 2**l, res[1] // 2**l, res[2] // 2**l]
            )
            for l in range(self.n_mg_levels)
        ]

        self.Adiag = [
            ti.field(
                dtype=ti.f32, shape=[res[0] // 2**l, res[1] // 2**l, res[2] // 2**l]
            )
            for l in range(self.n_mg_levels)
        ]  # A(i,j,k)(i,j,k)

        self.Ax = [
            ti.Vector.field(
                3,
                dtype=ti.f32,
                shape=[res[0] // 2**l, res[1] // 2**l, res[2] // 2**l],
            )
            for l in range(self.n_mg_levels)
        ]  # Ax=A(i,j,k)(i+1,j,k), Ay=A(i,j,k)(i,j+1,k), Az=A(i,j,k)(i,j,k+1)

        # cg
        self.x = ti.field(dtype=real, shape=res)  # solution
        self.p = ti.field(dtype=real, shape=res)  # conjugate gradient
        self.Ap = ti.field(dtype=real, shape=res)  # matrix-vector product
        self.sum = ti.field(dtype=real, shape=())  # storage for reductions
        self.alpha = ti.field(dtype=real, shape=())  # step size
        self.beta = ti.field(dtype=real, shape=())  # step size

        self.gs_buf = [
            ti.Vector.field(2, ti.f32, shape=[res[0] // 2**l, res[1] // 2**l, res[2] // 2**l])
            for l in range(self.n_mg_levels)
        ]
        
        self.H1 = [
            ti.Matrix.field(3, 2, dtype=ti.f32, shape=[res[0] // 2**l, res[1] // 2**l, res[2] // 2**l])
            for l in range(1, self.n_mg_levels - 1)
        ]
        self.H2 = [
            ti.Matrix.field(3, 2, dtype=ti.f32, shape=[res[0] // 2**l, res[1] // 2**l, res[2] // 2**l])
            for l in range(1, self.n_mg_levels - 1)
        ]
        self.H3 = [
            ti.Matrix.field(3, 2, dtype=ti.f32, shape=[res[0] // 2**l, res[1] // 2**l, res[2] // 2**l])
            for l in range(1, self.n_mg_levels - 1)
        ]
        self.H4 = [
            ti.Matrix.field(3, 2, dtype=ti.f32, shape=[res[0] // 2**l, res[1] // 2**l, res[2] // 2**l])
            for l in range(1, self.n_mg_levels - 1)
        ]

        self.residual_check_buffer = ti.field(dtype=real, shape=res)
        self.residual_list = []

        self.hidden_dim = hidden_dim
        self.layer1_para = [
            [ti.field(shape=(self.hidden_dim, 4), dtype=ti.f32) for _ in range(4)]
            for l in range(1, self.n_mg_levels - 1)
        ]
        self.layer2_para = [
            [ti.field(shape=(self.hidden_dim + 1, ), dtype=ti.f32) for _ in range(4)]
            for l in range(1, self.n_mg_levels - 1)
        ]
        self.R_para = [
            ti.field(shape=(2, ), dtype= ti.f32)
            for l in range(1, self.n_mg_levels - 1)
        ]
        # Note that we still have some running-time parameters.
        self.runtime_para = {
            'P_H_data': [[ti.field(shape=(2, ), dtype=ti.f32) for _ in range(4)] for l in range(1, self.n_mg_levels - 1)],
            'P_R_data': [ti.field(shape=(2, ), dtype=ti.f32) for l in range(1, self.n_mg_levels - 1)],
            'alpha': [ti.field(shape=(1, ), dtype=ti.f32) for _ in range(6)]
        }

    def load_para_to_taichi_field(self, nn_para):
        # For now we interpret the nn_para in the following order.
        accumulated_index = 0
        # [precompute_para], [runtime_para]
        for l in range(1, self.n_mg_levels - 1):
            # [layer1_para], [layer2_para], [R_para]
            for i in range(4):
                # Taichi field does not support assignment. Use from_numpy instead.
                self.layer1_para[l - 1][i].from_numpy(nn_para[accumulated_index: accumulated_index + self.hidden_dim * 4].reshape(self.hidden_dim, 4))
                accumulated_index += self.hidden_dim * 4
                self.layer2_para[l - 1][i].from_numpy(nn_para[accumulated_index: accumulated_index + self.hidden_dim + 1])
                accumulated_index += self.hidden_dim + 1
            self.R_para[l - 1].from_numpy(nn_para[accumulated_index: accumulated_index + 2])
            accumulated_index += 2
        # [P_H_data], [P_R_data], [alpha]
        for l in range(1, self.n_mg_levels - 1):
            # [P_H_data]
            for i in range(4):
                self.runtime_para['P_H_data'][l - 1][i].from_numpy(nn_para[accumulated_index: accumulated_index + 2])
                accumulated_index += 2
        for l in range(1, self.n_mg_levels - 1):
            # [P_R_data]
            self.runtime_para['P_R_data'][l - 1].from_numpy(nn_para[accumulated_index: accumulated_index + 2])
            accumulated_index += 2
        for i in range(6):
            self.runtime_para['alpha'][i].from_numpy(np.exp(nn_para[accumulated_index: accumulated_index + 1]) + 1)
            accumulated_index += 1
        assert accumulated_index == len(nn_para), "The length of nn_para is not equal to the accumulated_index."

    def kaiming_init_para(self):
        for l in range(1, self.n_mg_levels - 1):
            for i in range(4):
                self.layer1_para[l - 1][i].from_numpy(np.random.randn(self.hidden_dim, 4) * np.sqrt(2 / 4))
                self.layer2_para[l - 1][i].fill(0.0)
            self.R_para[l - 1].fill(0.0)
        for l in range(1, self.n_mg_levels - 1):
            for i in range(4):
                self.runtime_para['P_H_data'][l - 1][i].fill(0.0)
        for l in range(1, self.n_mg_levels - 1):
            self.runtime_para['P_R_data'][l - 1].fill(0.0)
        for i in range(6):
            self.runtime_para['alpha'][i].fill(2.0)

    @ti.kernel
    def check_residual(self):
        for I in ti.grouped(self.residual_check_buffer):
            if self.is_dof[0][I]:
                r = self.Adiag[0][I] * self.x[I]
                r += self.neighbor_sum(self.is_dof[0], self.Ax[0], self.x, I)
                self.residual_check_buffer[I] = r - self.b[I]
            else:
                self.residual_check_buffer[I] = ti.cast(0.0, self.real)

    @ti.kernel
    def nn_precompute(
        self,
        is_dof: ti.template(),
        Adiag: ti.template(),
        Ax: ti.template(),
        H: ti.template(),
        layer1_para: ti.template(),
        layer2_para: ti.template(),
        R_para: ti.template(),
        phase: ti.template(),
        hidden_dim: ti.template()
    ):
        for I in ti.grouped(is_dof):
            if (I.sum() & 1) == phase and is_dof[I]:
                # Gather the coefficients.
                u001 = Ax[I - [1, 0, 0]][0] if I[0] > 0 else ti.cast(0.0, ti.f32)
                u010 = Ax[I - [0, 1, 0]][1] if I[1] > 0 else ti.cast(0.0, ti.f32)
                u100 = Ax[I - [0, 0, 1]][2] if I[2] > 0 else ti.cast(0.0, ti.f32)
                v001 = Ax[I][0] if I[0] < Adiag.shape[0] - 1 else ti.cast(0.0, ti.f32)
                v010 = Ax[I][1] if I[1] < Adiag.shape[1] - 1 else ti.cast(0.0, ti.f32)
                v100 = Ax[I][2] if I[2] < Adiag.shape[2] - 1 else ti.cast(0.0, ti.f32)
                a_sum = u001 + u010 + u100 + v001 + v010 + v100
                u = ti.Vector([u001, u010, u100])
                v = ti.Vector([v001, v010, v100])
                # Assemble C and added to the H.
                result_H = ti.Matrix([
                    [u001, u010, u100],
                    [v001, v010, v100]
                ])

                for i in ti.static(range(hidden_dim)):
                    for j in ti.static(range(3)):
                        c = layer1_para[i, 2] * a_sum + layer1_para[i, 3]
                        result_H[0, j] += layer2_para[i] * ti.tanh(layer1_para[i, 0] * u[j] + layer1_para[i, 1] * v[j] + c)
                        result_H[1, j] += layer2_para[i] * ti.tanh(layer1_para[i, 1] * u[j] + layer1_para[i, 0] * v[j] + c)

                # Now we write H back.
                for i in ti.static(range(3)):
                    if I[i] > 0:
                        H[I - ti.Vector.unit(3, i)][i, 0] = u[i] + u[i] * R_para[0] + R_para[1]
                        H[I - ti.Vector.unit(3, i)][i, 1] = result_H[0, i] + layer2_para[hidden_dim]
                    if I[i] < Adiag.shape[i] - 1:
                        H[I][i, 0] = v[i] + v[i] * R_para[0] + R_para[1]
                        H[I][i, 1] = result_H[1, i] + layer2_para[hidden_dim]
    
    def v_cycle_all(self, name, timing=False):
        if timing:
            ti.sync()
            start_time = time.perf_counter_ns()
        if name == "original":
            self.v_cycle()
        elif name == "dual":
            self.v_cycle_gs_buf()
        elif name == "nn":
            self.v_cycle_nn()
        if timing:
            ti.sync()
            end_time = time.perf_counter_ns()
            print(f"V-cycle {name} time: {(end_time - start_time) / 1e6:.4f}ms")

    def solve(
        self,
        max_iters=-1,
        verbose=False,
        rel_tol_sqr=1e-12,
        eps=1e-20,
        skip_conv_check=False,
        v_cycle_name="original",
        v_cycle_timing=False,
        collect_residual=False
    ):
        # start from zero initial guess
        self.x.fill(0)
        self.r[0].copy_from(self.b)

        # compute initial residual and tolerance
        self.reduce(self.r[0], self.r[0])
        initial_rTr = self.sum[None]
        if verbose:
            print(f"init |residual|_2 = {ti.sqrt(initial_rTr)}")
        if collect_residual:
            self.residual_list = [ti.sqrt(initial_rTr)]
        tol = initial_rTr * rel_tol_sqr

        self.v_cycle_all(v_cycle_name, v_cycle_timing)
        self.p.copy_from(self.z[0])
        self.reduce(self.z[0], self.r[0])
        old_zTr = self.sum[None]

        # main loop
        iter = 0
        while max_iters == -1 or iter < max_iters:
            self.compute_Ap()
            self.reduce(self.p, self.Ap)
            pAp = self.sum[None]
            self.alpha[None] = old_zTr / (pAp + eps)

            self.update_xr()
            
            if not skip_conv_check:
                # check for convergence
                self.reduce(self.r[0], self.r[0])
                rTr = self.sum[None]

                if verbose:
                    print(f"iter {iter}, |residual|_2={ti.sqrt(rTr)}")

                if collect_residual:
                    self.check_residual()
                    self.reduce(self.residual_check_buffer, self.residual_check_buffer)
                    residual_norm = self.sum[None]
                    self.residual_list.append(ti.sqrt(residual_norm))

                if rTr < tol:
                    break

            self.v_cycle_all(v_cycle_name, v_cycle_timing)

            self.reduce(self.z[0], self.r[0])
            new_zTr = self.sum[None]
            self.beta[None] = new_zTr / (old_zTr + eps)

            self.update_p()
            old_zTr = new_zTr

            iter += 1
        
        return iter

    def build(self):
        self.build_multigrid()

    def build_multigrid(self):
        for l in range(1, self.n_mg_levels):
            self.coarsen_kernel(
                self.is_dof[l - 1],
                self.is_dof[l],
                self.Adiag[l - 1],
                self.Adiag[l],
                self.Ax[l - 1],
                self.Ax[l],
            )
    
    def build_nn(self):
        for l in range(1, self.n_mg_levels - 1):
            for h, H in enumerate([self.H1, self.H2, self.H3, self.H4]):
                self.nn_precompute(
                    self.is_dof[l],
                    self.Adiag[l],
                    self.Ax[l],
                    H[l - 1],
                    self.layer1_para[l - 1][h],
                    self.layer2_para[l - 1][h],
                    self.R_para[l - 1],
                    h % 2,
                    self.hidden_dim
                )

    @ti.func
    def get_offset(self, k):
        ret = ti.Vector([k % 2, (k // 2) % 2, k // 4])
        return ret

    @ti.func
    def cover(self, I, J):
        return all(J >= 2 * I) and all(J < 2 * I + 2)

    @ti.kernel
    def coarsen_kernel(
        self,
        fine_is_dof: ti.template(),
        coarse_is_dof: ti.template(),
        fine_Adiag: ti.template(),
        coarse_Adiag: ti.template(),
        fine_Ax: ti.template(),
        coarse_Ax: ti.template(),
    ):
        # is_dof
        for I in ti.grouped(coarse_is_dof):
            base_fine_coord = I * 2
            is_dof_ret = False
            for k in ti.static(range(8)):
                offset = self.get_offset(k)
                fine_coord = base_fine_coord + offset
                is_dof_ret |= fine_is_dof[fine_coord]
            coarse_is_dof[I] = is_dof_ret

        # Adiag
        for I in ti.grouped(coarse_Adiag):
            Adiag_ret = ti.cast(0.0, ti.f32)
            if coarse_is_dof[I]:
                base_fine_coord = I * 2
                for k in ti.static(range(8)):
                    offset = self.get_offset(k)
                    fine_coord = base_fine_coord + offset
                    if fine_is_dof[fine_coord]:
                        Adiag_ret += fine_Adiag[fine_coord]
                        for i in ti.static(range(3)):
                            nb_fine_coord = fine_coord + ti.Vector.unit(3, i)
                            if (
                                all(nb_fine_coord < fine_is_dof.shape)
                                and fine_is_dof[nb_fine_coord]
                                and self.cover(I, nb_fine_coord)
                            ):
                                Adiag_ret += (
                                    ti.cast(2.0, ti.f32) * fine_Ax[fine_coord][i]
                                )
                Adiag_ret *= ti.cast(0.25, ti.f32)
            coarse_Adiag[I] = Adiag_ret

        # Ax
        for I in ti.grouped(coarse_Ax):
            Ax_ret = ti.Vector.zero(n=3, dt=ti.f32)
            if coarse_is_dof[I]:
                base_fine_coord = I * 2
                for k in ti.static([1, 3, 5, 7]):
                    offset = self.get_offset(k)
                    fine_coord = base_fine_coord + offset
                    if fine_is_dof[fine_coord]:
                        nb_fine_coord = fine_coord + ti.Vector.unit(3, 0)
                        if (
                            all(nb_fine_coord < fine_is_dof.shape)
                            and fine_is_dof[nb_fine_coord]
                        ):
                            Ax_ret[0] += fine_Ax[fine_coord][0]
                for k in ti.static([2, 3, 6, 7]):
                    offset = self.get_offset(k)
                    fine_coord = base_fine_coord + offset
                    if fine_is_dof[fine_coord]:
                        nb_fine_coord = fine_coord + ti.Vector.unit(3, 1)
                        if (
                            all(nb_fine_coord < fine_is_dof.shape)
                            and fine_is_dof[nb_fine_coord]
                        ):
                            Ax_ret[1] += fine_Ax[fine_coord][1]
                for k in ti.static([4, 5, 6, 7]):
                    offset = self.get_offset(k)
                    fine_coord = base_fine_coord + offset
                    if fine_is_dof[fine_coord]:
                        nb_fine_coord = fine_coord + ti.Vector.unit(3, 2)
                        if (
                            all(nb_fine_coord < fine_is_dof.shape)
                            and fine_is_dof[nb_fine_coord]
                        ):
                            Ax_ret[2] += fine_Ax[fine_coord][2]
                Ax_ret *= ti.cast(0.25, ti.f32)
            coarse_Ax[I] = Ax_ret

    @ti.kernel
    def prolongate(self, l: ti.template()):
        for I in ti.grouped(self.z[l]):
            if self.is_dof[l][I]:
                self.z[l][I] += (
                    self.z[l + 1][I // 2] + self.z[l + 1][I // 2]
                )  # 2.0 for fast convergence

    @ti.func
    def neighbor_sum(self, is_dof, Ax, x, I):
        ret = ti.cast(0.0, x.dtype)
        for i in ti.static(range(3)):
            offset = ti.Vector.unit(3, i)
            if (
                all(I - offset >= 0)
                and all(I - offset < x.shape)
                and is_dof[I - offset]
            ):
                ret += Ax[I - offset][i] * x[I - offset]
            if (
                all(I + offset >= 0)
                and all(I + offset < x.shape)
                and is_dof[I + offset]
            ):
                ret += Ax[I][i] * x[I + offset]
        return ret

    @ti.kernel
    def restrict(self, l: ti.template()):
        for I in ti.grouped(self.r[l + 1]):
            if self.is_dof[l + 1][I]:
                base_fine_coord = I * 2
                ret = 0.0
                for k in ti.static(range(8)):
                    offset = self.get_offset(k)
                    fine_coord = base_fine_coord + offset
                    if self.is_dof[l][fine_coord]:
                        Az = self.Adiag[l][fine_coord] * self.z[l][fine_coord]
                        Az += self.neighbor_sum(
                            self.is_dof[l], self.Ax[l], self.z[l], fine_coord
                        )
                        ret += self.r[l][fine_coord] - Az
                self.r[l + 1][I] = ret * ti.cast(0.25, ti.f32)
            else:
                self.r[l + 1][I] = ti.cast(0.0, ti.f32)

    @ti.kernel
    def reduce(self, p: ti.template(), q: ti.template()):
        self.sum[None] = ti.cast(0.0, self.sum.dtype)
        for I in ti.grouped(p):
            if self.is_dof[0][I]:
                self.sum[None] += p[I] * q[I]

    @ti.kernel
    def compute_Ap(self):
        for I in ti.grouped(self.Ap):
            if self.is_dof[0][I]:
                r = self.Adiag[0][I] * self.p[I]
                r += self.neighbor_sum(self.is_dof[0], self.Ax[0], self.p, I)
                self.Ap[I] = r
            else:
                self.Ap[I] = ti.cast(0.0, self.real)

    @ti.kernel
    def update_xr(self):
        alpha = self.alpha[None]
        for I in ti.grouped(self.p):
            if self.is_dof[0][I]:
                self.x[I] += alpha * self.p[I]
                self.r[0][I] -= alpha * self.Ap[I]

    @ti.kernel
    def update_p(self):
        for I in ti.grouped(self.p):
            if self.is_dof[0][I]:
                self.p[I] = self.z[0][I] + self.beta[None] * self.p[I]

    @ti.kernel
    def smooth(self, l: ti.template(), phase: ti.template()):
        # phase = red/black Gauss-Seidel phase
        for I in ti.grouped(self.r[l]):
            if (
                (I.sum()) & 1 == phase
                and self.is_dof[l][I]
                and self.Adiag[l][I] > ti.cast(0.0, ti.f32)
            ):
                self.z[l][I] = (
                    self.r[l][I]
                    - self.neighbor_sum(self.is_dof[l], self.Ax[l], self.z[l], I)
                ) / self.Adiag[l][I]


    def v_cycle(self):
        self.z[0].fill(0.0)
        for l in range(self.n_mg_levels - 1):
            self.smooth(l, 0)
            self.smooth(l, 1)
            self.smooth(l, 0)
            self.smooth(l, 1)
            self.z[l + 1].fill(0.0)
            self.restrict(l)

        # solve Az = r on the coarse grid
        for i in range(self.bottom_smoothing // 2):
            self.smooth(self.n_mg_levels - 1, 0)
            self.smooth(self.n_mg_levels - 1, 1)
        for i in range(self.bottom_smoothing // 2):
            self.smooth(self.n_mg_levels - 1, 1)
            self.smooth(self.n_mg_levels - 1, 0)

        for l in reversed(range(self.n_mg_levels - 1)):
            self.prolongate(l)
            self.smooth(l, 1)
            self.smooth(l, 0)
            self.smooth(l, 1)
            self.smooth(l, 0)

    # The implementation of the dual-channel V-cycle is here.
    @ti.kernel
    def smooth_gs_buf(self, l: ti.template(), phase: ti.template()):
        # phase = red/black Gauss-Seidel phase
        for I in ti.grouped(self.r[l]):
            if (
                (I.sum()) & 1 == phase
                and self.is_dof[l][I]
                and self.Adiag[l][I] > ti.cast(0.0, ti.f32)
            ):
                # Separately assemble channel 0 and channel 1.
                val0 = self.r[l][I]
                val1 = ti.cast(0.0, ti.f32)
                for i in ti.static(range(3)):
                    offset = ti.Vector.unit(3, i)
                    if (
                        all(I - offset >= 0)
                        and all(I - offset < self.r[l].shape)
                        and self.is_dof[l][I - offset]
                    ):
                        val0 -= self.Ax[l][I - offset][i] * self.gs_buf[l][I - offset][0]
                        val1 -= self.Ax[l][I - offset][i] * self.gs_buf[l][I - offset][1]
                    if (
                        all(I + offset >= 0)
                        and all(I + offset < self.r[l].shape)
                        and self.is_dof[l][I + offset]
                    ):
                        val0 -= self.Ax[l][I][i] * self.gs_buf[l][I + offset][0]
                        val1 -= self.Ax[l][I][i] * self.gs_buf[l][I + offset][1]
                self.gs_buf[l][I] = [val0 / self.Adiag[l][I], val1 / self.Adiag[l][I]]

    @ti.kernel
    def smooth_gs_buf_transpose_first(self, l: ti.template()):
        # phase = red/black Gauss-Seidel phase
        for I in ti.grouped(self.r[l]):
            if (
                (I.sum()) & 1 == 0
                and self.is_dof[l][I]
                and self.Adiag[l][I] > ti.cast(0.0, ti.f32)
            ):
                self.gs_buf[l][I][0] = self.r[l][I] / self.Adiag[l][I]
    
    @ti.kernel
    def smooth_gs_buf_transpose_second(self, l: ti.template()):
        for I in ti.grouped(self.r[l]):
            if (
                (I.sum()) & 1 == 1
                and self.is_dof[l][I]
                and self.Adiag[l][I] > ti.cast(0.0, ti.f32)
            ):
                # Separately assemble channel 0 and channel 1.
                val0 = self.r[l][I]
                val1 = ti.cast(0.0, ti.f32)
                for i in ti.static(range(3)):
                    offset = ti.Vector.unit(3, i)
                    if (
                        all(I - offset >= 0)
                        and all(I - offset < self.r[l].shape)
                        and self.is_dof[l][I - offset]
                    ):
                        val0 -= self.Ax[l][I - offset][i] * self.gs_buf[l][I - offset][0]
                        val1 -= self.Ax[l][I - offset][i] * self.gs_buf[l][I - offset][0]
                    if (
                        all(I + offset >= 0)
                        and all(I + offset < self.r[l].shape)
                        and self.is_dof[l][I + offset]
                    ):
                        val0 -= self.Ax[l][I][i] * self.gs_buf[l][I + offset][0]
                        val1 -= self.Ax[l][I][i] * self.gs_buf[l][I + offset][0]
                val1 += self.r[l][I]
                self.gs_buf[l][I] = [val0 / self.Adiag[l][I], val1]

    @ti.kernel
    def smooth_gs_buf_transpose(self, l: ti.template(), phase: ti.template()):
        # phase = red/black Gauss-Seidel phase
        for I in ti.grouped(self.r[l]):
            if (
                (I.sum()) & 1 == phase
                and self.is_dof[l][I]
                and self.Adiag[l][I] > ti.cast(0.0, ti.f32)
            ):
                # Separately assemble channel 0 and channel 1.
                val0 = self.r[l][I]
                val1 = ti.cast(0.0, ti.f32)
                for i in ti.static(range(3)):
                    offset = ti.Vector.unit(3, i)
                    if (
                        all(I - offset >= 0)
                        and all(I - offset < self.r[l].shape)
                        and self.is_dof[l][I - offset]
                    ):
                        val0 -= self.Ax[l][I - offset][i] * self.gs_buf[l][I - offset][0]
                        val1 -= self.Ax[l][I - offset][i] * self.gs_buf[l][I - offset][1] / self.Adiag[l][I - offset]
                    if (
                        all(I + offset >= 0)
                        and all(I + offset < self.r[l].shape)
                        and self.is_dof[l][I + offset]
                    ):
                        val0 -= self.Ax[l][I][i] * self.gs_buf[l][I + offset][0]
                        val1 -= self.Ax[l][I][i] * self.gs_buf[l][I + offset][1] / self.Adiag[l][I + offset]
                self.gs_buf[l][I] = [val0 / self.Adiag[l][I], val1]

    @ti.kernel
    def restrict_gs_buf(self, l: ti.template()):
        # In our restrict, we simply coarsen the gs_buf[l][1] to r[l + 1], and no Az is involved.
        for I in ti.grouped(self.gs_buf[l + 1]):
            if self.is_dof[l + 1][I]:
                base_fine_coord = I * 2
                ret = 0.0
                for k in ti.static(range(8)):
                    offset = self.get_offset(k)
                    if offset.sum() & 1 == 0:
                        fine_coord = base_fine_coord + offset
                        if self.is_dof[l][fine_coord]:
                            # ret += self.gs_buf[l][fine_coord][1]
                            for i in ti.static(range(3)):
                                offset2 = ti.Vector.unit(3, i)
                                if (
                                    all(fine_coord - offset2 >= 0)
                                    and all(fine_coord - offset2 < self.r[l].shape)
                                    and self.is_dof[l][fine_coord - offset2]
                                ):
                                    ret -= self.Ax[l][fine_coord - offset2][i] * self.gs_buf[l][fine_coord - offset2][1] / self.Adiag[l][fine_coord - offset2]
                                if (
                                    all(fine_coord + offset2 >= 0)
                                    and all(fine_coord + offset2 < self.r[l].shape)
                                    and self.is_dof[l][fine_coord + offset2]
                                ):
                                    ret -= self.Ax[l][fine_coord][i] * self.gs_buf[l][fine_coord + offset2][1] / self.Adiag[l][fine_coord + offset2]
                self.r[l + 1][I] = ret * ti.cast(0.25, ti.f32)
            else:
                self.r[l + 1][I] = ti.cast(0.0, ti.f32)
    
    @ti.kernel
    def add_to_z(self, l: ti.template()):
        for I in ti.grouped(self.z[l]):
            if self.is_dof[l][I]:
                val = self.gs_buf[l][I].sum()
                self.z[l][I] = val

    @ti.kernel
    def prolongate_gs_buf(self, l: ti.template()):
        for I in ti.grouped(self.gs_buf[l]):
            if self.is_dof[l][I]:
                self.gs_buf[l][I][1] = (
                    ti.cast(2.0, ti.f32) * self.z[l + 1][I // 2]
                )

    @ti.kernel
    def copy_bottom_level_to_gs_buf(self):
        for I in ti.grouped(self.gs_buf[-1]):
            self.gs_buf[-1][I][1] = self.z[-1][I]
    
    def v_cycle_gs_buf(self):
        # Only change the computational structure, but do not introduce the nn parameters.
        self.z[0].fill(0.0)
        self.smooth(0, 0)
        self.smooth(0, 1)
        self.smooth(0, 0)
        self.smooth(0, 1)
        self.restrict(0)
        start_l = 1
        for l in range(start_l, self.n_mg_levels - 1):
            for i in range(self.pre_and_post_smoothing):
                if i == 0:
                    self.smooth_gs_buf_transpose_first(l)
                    self.smooth_gs_buf_transpose_second(l)
                else:
                    self.smooth_gs_buf_transpose(l, 0)
                    self.smooth_gs_buf_transpose(l, 1)
            self.restrict_gs_buf(l)
        self.z[self.n_mg_levels - 1].fill(0.0)
        for i in range(self.bottom_smoothing // 2):
            self.smooth(self.n_mg_levels - 1, 0)
            self.smooth(self.n_mg_levels - 1, 1)
        for i in range(self.bottom_smoothing // 2):
            self.smooth(self.n_mg_levels - 1, 1)
            self.smooth(self.n_mg_levels - 1, 0)
        # Now the result is stored in z.
        self.copy_bottom_level_to_gs_buf()
        # The solved things are stored into z's.
        for l in reversed(range(start_l, self.n_mg_levels - 1)):
            self.prolongate_gs_buf(l)
            self.smooth_gs_buf(l, 1)
            self.smooth_gs_buf(l, 0)
            self.smooth_gs_buf(l, 1)
            self.smooth_gs_buf(l, 0)
            self.add_to_z(l)
        self.prolongate(0)
        self.smooth(0, 1)
        self.smooth(0, 0)
        self.smooth(0, 1)
        self.smooth(0, 0)
    
    @ti.func
    def P_nn(self, I, A, nn_data):
        return ti.cast(1.0, ti.f32) / (A[I] + nn_data[0]) + nn_data[1]
    
    # NN versions:
    # Smoothers.
    @ti.kernel
    def smooth_gs_buf_nn(self, l: ti.template(), phase: ti.template(), H: ti.template(), P_R_data: ti.template(), P_H_data: ti.template()):
        for I in ti.grouped(self.r[l]):
            if (
                (I.sum()) & 1 == phase
                and self.is_dof[l][I]
                and self.Adiag[l][I] > ti.cast(0.0, ti.f32)
            ):
                # Separately assemble channel 0 and channel 1.
                val0 = self.r[l][I]
                val1 = ti.cast(0.0, ti.f32)
                for i in ti.static(range(3)):
                    offset = ti.Vector.unit(3, i)
                    if (
                        all(I - offset >= 0)
                        and all(I - offset < self.r[l].shape)
                        and self.is_dof[l][I - offset]
                    ):
                        val0 -= H[I - offset][i, 0] * self.gs_buf[l][I - offset][0]
                        val1 -= H[I - offset][i, 1] * self.gs_buf[l][I - offset][1]
                    if (
                        all(I + offset >= 0)
                        and all(I + offset < self.r[l].shape)
                        and self.is_dof[l][I + offset]
                    ):
                        val0 -= H[I][i, 0] * self.gs_buf[l][I + offset][0]
                        val1 -= H[I][i, 1] * self.gs_buf[l][I + offset][1]
                self.gs_buf[l][I] = [val0 * self.P_nn(I, self.Adiag[l], P_R_data), val1 * self.P_nn(I, self.Adiag[l], P_H_data)]
    
    # Transpose versions.
    @ti.kernel
    def smooth_gs_buf_transpose_first_nn(self, l: ti.template(), P_R_data: ti.template(), P_H_data: ti.template()):
        for I in ti.grouped(self.r[l]):
            if (
                (I.sum()) & 1 == 0
                and self.is_dof[l][I]
                and self.Adiag[l][I] > ti.cast(0.0, ti.f32)
            ):
                self.gs_buf[l][I] = [self.r[l][I] * self.P_nn(I, self.Adiag[l], P_R_data), self.r[l][I] * self.P_nn(I, self.Adiag[l], P_H_data)]

    @ti.kernel
    def smooth_gs_buf_transpose_second_nn(self, l: ti.template(), H: ti.template(), P_R_data: ti.template()):
        for I in ti.grouped(self.r[l]):
            if (
                (I.sum()) & 1 == 1
                and self.is_dof[l][I]
                and self.Adiag[l][I] > ti.cast(0.0, self.real)
            ):
                # Separately assemble channel 0 and channel 1.
                val0 = self.r[l][I]
                val1 = val0
                for i in ti.static(range(3)):
                    offset = ti.Vector.unit(3, i)
                    if (
                        all(I - offset >= 0)
                        and all(I - offset < self.r[l].shape)
                        and self.is_dof[l][I - offset]
                    ):
                        val0 -= H[I - offset][i, 0] * self.gs_buf[l][I - offset][0]
                        val1 -= H[I - offset][i, 1] * self.gs_buf[l][I - offset][1]
                    if (
                        all(I + offset >= 0)
                        and all(I + offset < self.r[l].shape)
                        and self.is_dof[l][I + offset]
                    ):
                        val0 -= H[I][i, 0] * self.gs_buf[l][I + offset][0]
                        val1 -= H[I][i, 1] * self.gs_buf[l][I + offset][1]
                self.gs_buf[l][I] = [self.P_nn(I, self.Adiag[l], P_R_data) * val0, val1]
    
    @ti.kernel
    def smooth_gs_buf_transpose_nn(self, l: ti.template(), phase: ti.template(), H: ti.template(), P_R_data: ti.template(), P_H_data: ti.template()):
        for I in ti.grouped(self.r[l]):
            if (
                (I.sum()) & 1 == phase
                and self.is_dof[l][I]
                and self.Adiag[l][I] > ti.cast(0.0, ti.f32)
            ):
                # Separately assemble channel 0 and channel 1.
                val0 = self.r[l][I]
                val1 = ti.cast(0.0, ti.f32)
                for i in ti.static(range(3)):
                    offset = ti.Vector.unit(3, i)
                    if (
                        all(I - offset >= 0)
                        and all(I - offset < self.r[l].shape)
                        and self.is_dof[l][I - offset]
                    ):
                        val0 -= H[I - offset][i, 0] * self.gs_buf[l][I - offset][0]
                        val1 -= H[I - offset][i, 1] * self.gs_buf[l][I - offset][1] * self.P_nn(I - offset, self.Adiag[l], P_H_data)
                    if (
                        all(I + offset >= 0)
                        and all(I + offset < self.r[l].shape)
                        and self.is_dof[l][I + offset]
                    ):
                        val0 -= H[I][i, 0] * self.gs_buf[l][I + offset][0]
                        val1 -= H[I][i, 1] * self.gs_buf[l][I + offset][1] * self.P_nn(I + offset, self.Adiag[l], P_H_data)
                self.gs_buf[l][I] = [val0 * self.P_nn(I, self.Adiag[l], P_R_data), val1]

    # Restrict and prolongate.
    @ti.kernel
    def restrict_gs_buf_nn(self, l: ti.template(), H: ti.template(), P_H_data: ti.template()):
        for I in ti.grouped(self.gs_buf[l + 1]):
            if self.is_dof[l + 1][I]:
                base_fine_coord = I * 2
                ret = 0.0
                for k in ti.static(range(8)):
                    offset = self.get_offset(k)
                    if offset.sum() & 1 == 0:
                        fine_coord = base_fine_coord + offset
                        if self.is_dof[l][fine_coord]:
                            for i in ti.static(range(3)):
                                offset2 = ti.Vector.unit(3, i)
                                if (
                                    all(fine_coord - offset2 >= 0)
                                    and all(fine_coord - offset2 < self.r[l].shape)
                                    and self.is_dof[l][fine_coord - offset2]
                                ):
                                    ret -= H[fine_coord - offset2][i, 1] * self.gs_buf[l][fine_coord - offset2][1] * self.P_nn(fine_coord - offset2, self.Adiag[l], P_H_data)
                                if (
                                    all(fine_coord + offset2 >= 0)
                                    and all(fine_coord + offset2 < self.r[l].shape)
                                    and self.is_dof[l][fine_coord + offset2]
                                ):
                                    ret -= H[fine_coord][i, 1] * self.gs_buf[l][fine_coord + offset2][1] * self.P_nn(fine_coord + offset2, self.Adiag[l], P_H_data)
                self.r[l + 1][I] = ret * ti.cast(0.25, ti.f32)
            else:
                self.r[l + 1][I] = ti.cast(0.0, ti.f32)

    @ti.kernel
    def prolongate_gs_buf_nn(self, l: ti.template(), alpha: ti.template()):
        for I in ti.grouped(self.gs_buf[l]):
            if self.is_dof[l][I]:
                self.gs_buf[l][I][1] = alpha[0] * self.z[l + 1][I // 2]

    @ti.kernel
    def polongate_gs_buf_top(self, alpha: ti.template()):
        for I in ti.grouped(self.z[0]):
            if self.is_dof[0][I]:
                self.z[0][I] += alpha[0] * self.z[1][I // 2]

    def v_cycle_nn(self):
        self.z[0].fill(0.0)
        self.smooth(0, 0)
        self.smooth(0, 1)
        self.smooth(0, 0)
        self.smooth(0, 1)
        self.restrict(0)
        start_l = 1
        for l in range(start_l, self.n_mg_levels - 1):
            self.smooth_gs_buf_transpose_first_nn(l, self.runtime_para['P_R_data'][l - start_l], self.runtime_para['P_H_data'][l - start_l][0])
            self.smooth_gs_buf_transpose_second_nn(l, self.H1[l - start_l], self.runtime_para['P_R_data'][l - start_l])
            self.smooth_gs_buf_transpose_nn(l, 0, self.H2[l - start_l], self.runtime_para['P_R_data'][l - start_l], self.runtime_para['P_H_data'][l - start_l][1])
            self.smooth_gs_buf_transpose_nn(l, 1, self.H3[l - start_l], self.runtime_para['P_R_data'][l - start_l], self.runtime_para['P_H_data'][l - start_l][2])
            self.restrict_gs_buf_nn(l, self.H4[l - start_l], self.runtime_para['P_H_data'][l - start_l][3])
        self.z[self.n_mg_levels - 1].fill(0.0)
        for i in range(self.bottom_smoothing // 2):
            self.smooth(self.n_mg_levels - 1, 0)
            self.smooth(self.n_mg_levels - 1, 1)
        for i in range(self.bottom_smoothing // 2):
            self.smooth(self.n_mg_levels - 1, 1)
            self.smooth(self.n_mg_levels - 1, 0)
        # Now the result is stored in z.
        self.copy_bottom_level_to_gs_buf()
        # The solved things are stored into z's.
        for l in reversed(range(start_l, self.n_mg_levels - 1)):
            # You should reload the runtime parameters here!
            self.prolongate_gs_buf_nn(l, self.runtime_para['alpha'][l])
            self.smooth_gs_buf_nn(l, 1, self.H4[l - start_l], self.runtime_para['P_R_data'][l - start_l], self.runtime_para['P_H_data'][l - start_l][3])
            self.smooth_gs_buf_nn(l, 0, self.H3[l - start_l], self.runtime_para['P_R_data'][l - start_l], self.runtime_para['P_H_data'][l - start_l][2])
            self.smooth_gs_buf_nn(l, 1, self.H2[l - start_l], self.runtime_para['P_R_data'][l - start_l], self.runtime_para['P_H_data'][l - start_l][1])
            self.smooth_gs_buf_nn(l, 0, self.H1[l - start_l], self.runtime_para['P_R_data'][l - start_l], self.runtime_para['P_H_data'][l - start_l][0])
            self.add_to_z(l)
        self.polongate_gs_buf_top(self.runtime_para['alpha'][0])
        self.smooth(0, 1)
        self.smooth(0, 0)
        self.smooth(0, 1)
        self.smooth(0, 0)
