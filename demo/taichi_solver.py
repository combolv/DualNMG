# A third-party Taichi UAAMG reference, copied from https://github.com/yuchen203/bubble/blob/main/solver.py.
import taichi as ti
import numpy as np
import math

@ti.data_oriented
class AMGPCG_2D:
    def __init__(self, res, base_level=3, real=float):
        # parameters
        self.res = res
        self.n_mg_levels = int(math.log2(min(res))) - base_level + 1
        self.real = real

        self.pre_and_post_smoothing = 2
        self.bottom_smoothing = 10

        # rhs
        self.b = ti.field(dtype=real, shape=res)  # Ax=b

        self.r = [
            ti.field(dtype=real, shape=[res[0] // 2**l, res[1] // 2**l])
            for l in range(self.n_mg_levels)
        ]  # residual
        self.z = [
            ti.field(dtype=real, shape=[res[0] // 2**l, res[1] // 2**l])
            for l in range(self.n_mg_levels)
        ]  # M^-1 self.r

        # lhs
        self.is_dof = [
            ti.field(dtype=bool, shape=[res[0] // 2**l, res[1] // 2**l])
            for l in range(self.n_mg_levels)
        ]

        self.Adiag = [
            ti.field(dtype=real, shape=[res[0] // 2**l, res[1] // 2**l])
            for l in range(self.n_mg_levels)
        ]  # A(i,j,k)(i,j,k)

        self.Ax = [
            ti.Vector.field(2, dtype=real, shape=[res[0] // 2**l, res[1] // 2**l])
            for l in range(self.n_mg_levels)
        ]  # Ax=A(i,j,k)(i+1,j,k), Ay=A(i,j,k)(i,j+1,k), Az=A(i,j,k)(i,j,k+1)

        # cg
        self.x = ti.field(dtype=real, shape=res)  # solution
        self.p = ti.field(dtype=real, shape=res)  # conjugate gradient
        self.Ap = ti.field(dtype=real, shape=res)  # matrix-vector product
        self.sum = ti.field(dtype=real, shape=())  # storage for reductions
        self.alpha = ti.field(dtype=real, shape=())  # step size
        self.beta = ti.field(dtype=real, shape=())  # step size

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

    def solve(
        self,
        pure_neumann=False,
        max_iters=-1,
        verbose=False,
        rel_tol=1e-12,
        abs_tol=1e-14,
        eps=1e-20
    ):

        # start from zero initial guess
        self.x.fill(0)
        self.r[0].copy_from(self.b)

        # compute initial residual and tolerance
        self.reduce(self.r[0], self.r[0])
        initial_rTr = self.sum[None]
        if verbose:
            print(f"init |residual|_2 = {ti.sqrt(initial_rTr)}")
        tol = max(abs_tol, initial_rTr * rel_tol)

        if pure_neumann:
            self.recenter(self.r[0])

        # set aux fields
        self.v_cycle()

        self.p.copy_from(self.z[0])
        self.reduce(self.z[0], self.r[0])
        old_zTr = self.sum[None]

        # main loop
        iter = 0
        while max_iters == -1 or iter < max_iters:

            # self.alpha = rTr / pTAp
            self.compute_Ap()
            self.reduce(self.p, self.Ap)
            pAp = self.sum[None]
            self.alpha[None] = old_zTr / (pAp + eps)

            # self.x = self.x + self.alpha self.p
            # self.r = self.r - self.alpha self.Ap
            self.update_xr()

            # check for convergence
            self.reduce(self.r[0], self.r[0])
            rTr = self.sum[None]

            if verbose:
                print(f"iter {iter}, |residual|_2={ti.sqrt(rTr)}")

            if rTr < tol:
                break

            if pure_neumann:
                self.recenter(self.r[0])

            self.v_cycle()

            # self.beta = new_rTr / old_rTr
            self.reduce(self.z[0], self.r[0])

            new_zTr = self.sum[None]
            self.beta[None] = new_zTr / (old_zTr + eps)

            # self.p = self.z + self.beta self.p
            self.update_p()
            old_zTr = new_zTr

            iter += 1


    @ti.func
    def get_offset(self, k):
        ret = ti.Vector([k % 2, k // 2])
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
            for k in ti.static(range(4)):
                offset = self.get_offset(k)
                fine_coord = base_fine_coord + offset
                is_dof_ret |= fine_is_dof[fine_coord]
            coarse_is_dof[I] = is_dof_ret

        # Adiag
        for I in ti.grouped(coarse_Adiag):
            Adiag_ret = 0.0
            if coarse_is_dof[I]:
                base_fine_coord = I * 2
                for k in ti.static(range(4)):
                    offset = self.get_offset(k)
                    fine_coord = base_fine_coord + offset
                    if fine_is_dof[fine_coord]:
                        Adiag_ret += fine_Adiag[fine_coord]
                        for i in ti.static(range(2)):
                            nb_fine_coord = fine_coord + ti.Vector.unit(2, i)
                            if (
                                all(nb_fine_coord < fine_is_dof.shape)
                                and fine_is_dof[nb_fine_coord]
                                and self.cover(I, nb_fine_coord)
                            ):
                                Adiag_ret += (
                                    ti.cast(2.0, self.real) * fine_Ax[fine_coord][i]
                                )
                Adiag_ret *= 0.25
            coarse_Adiag[I] = Adiag_ret

        # Ax
        for I in ti.grouped(coarse_Ax):
            Ax_ret = ti.Vector.zero(n=2, dt=self.real)
            if coarse_is_dof[I]:
                base_fine_coord = I * 2
                for k in ti.static([1, 3]):
                    offset = self.get_offset(k)
                    fine_coord = base_fine_coord + offset
                    if fine_is_dof[fine_coord]:
                        nb_fine_coord = fine_coord + ti.Vector.unit(2, 0)
                        if (
                            all(nb_fine_coord < fine_is_dof.shape)
                            and fine_is_dof[nb_fine_coord]
                        ):
                            Ax_ret[0] += fine_Ax[fine_coord][0]
                for k in ti.static([2, 3]):
                    offset = self.get_offset(k)
                    fine_coord = base_fine_coord + offset
                    if fine_is_dof[fine_coord]:
                        nb_fine_coord = fine_coord + ti.Vector.unit(2, 1)
                        if (
                            all(nb_fine_coord < fine_is_dof.shape)
                            and fine_is_dof[nb_fine_coord]
                        ):
                            Ax_ret[1] += fine_Ax[fine_coord][1]
                Ax_ret *= ti.cast(0.25, self.real)
            coarse_Ax[I] = Ax_ret

    @ti.kernel
    def reduce(self, p: ti.template(), q: ti.template()):
        self.sum[None] = ti.cast(0.0, self.real)
        for I in ti.grouped(p):
            if self.is_dof[0][I]:
                self.sum[None] += p[I] * q[I]

    @ti.kernel
    def update_p(self):
        for I in ti.grouped(self.p):
            if self.is_dof[0][I]:
                self.p[I] = self.z[0][I] + self.beta[None] * self.p[I]

    @ti.func
    def neighbor_sum(self, is_dof, Ax, x, I):
        ret = ti.cast(0.0, self.real)
        for i in ti.static(range(2)):
            offset = ti.Vector.unit(2, i)
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
    def restrict(self, l: ti.template()):
        for I in ti.grouped(self.r[l + 1]):
            if self.is_dof[l + 1][I]:
                base_fine_coord = I * 2
                ret = 0.0
                for k in ti.static(range(4)):
                    offset = self.get_offset(k)
                    fine_coord = base_fine_coord + offset
                    if self.is_dof[l][fine_coord]:
                        Az = self.Adiag[l][fine_coord] * self.z[l][fine_coord]
                        Az += self.neighbor_sum(
                            self.is_dof[l], self.Ax[l], self.z[l], fine_coord
                        )
                        ret += self.r[l][fine_coord] - Az
                self.r[l + 1][I] = ret * ti.cast(0.25, self.real)
            else:
                self.r[l + 1][I] = ti.cast(0.0, self.real)

    @ti.kernel
    def prolongate(self, l: ti.template()):
        for I in ti.grouped(self.z[l]):
            if self.is_dof[l][I]:
                self.z[l][I] += (
                    ti.cast(2.0, self.real) * self.z[l + 1][I // 2]
                )  # 2.0 for fast convergence

    @ti.kernel
    def smooth(self, l: ti.template(), phase: ti.template()):
        # phase = red/black Gauss-Seidel phase
        for I in ti.grouped(self.r[l]):
            if (
                (I.sum()) & 1 == phase
                and self.is_dof[l][I]
                and self.Adiag[l][I] > ti.cast(0.0, self.real)
            ):
                self.z[l][I] = (
                    self.r[l][I]
                    - self.neighbor_sum(self.is_dof[l], self.Ax[l], self.z[l], I)
                ) / self.Adiag[l][I]

    def v_cycle(self):
        self.z[0].fill(0.0)
        for l in range(self.n_mg_levels - 1):
            for i in range(self.pre_and_post_smoothing):
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
            for i in range(self.pre_and_post_smoothing):
                self.smooth(l, 1)
                self.smooth(l, 0)

    @ti.kernel
    def recenter(self, a: ti.template()):
        self.sum[None] = ti.cast(0.0, self.real)
        num_dof = 0.0
        for I in ti.grouped(a):
            if self.is_dof[0][I]:
                self.sum[None] += a[I]
                num_dof += 1.0
        mean = self.sum[None] / num_dof
        for I in ti.grouped(a):
            if self.is_dof[0][I]:
                a[I] -= mean

@ti.data_oriented
class AMGPCG_3D:
    def __init__(self, res, base_level=3, real=float):
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
                dtype=real, shape=[res[0] // 2**l, res[1] // 2**l, res[2] // 2**l]
            )
            for l in range(self.n_mg_levels)
        ]  # residual
        self.z = [
            ti.field(
                dtype=real, shape=[res[0] // 2**l, res[1] // 2**l, res[2] // 2**l]
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
                dtype=real, shape=[res[0] // 2**l, res[1] // 2**l, res[2] // 2**l]
            )
            for l in range(self.n_mg_levels)
        ]  # A(i,j,k)(i,j,k)

        self.Ax = [
            ti.Vector.field(
                3,
                dtype=real,
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

        # pure neumann
        self.num_dof = ti.field(dtype=int, shape=())

    def solve(
        self,
        pure_neumann=False,
        max_iters=-1,
        verbose=False,
        rel_tol=1e-12,
        abs_tol=1e-14,
        eps=1e-20,
    ):
        # start from zero initial guess
        self.x.fill(0)
        self.r[0].copy_from(self.b)

        # compute initial residual and tolerance
        self.reduce(self.r[0], self.r[0])
        initial_rTr = self.sum[None]
        if verbose:
            print(f"init |residual|_2 = {ti.sqrt(initial_rTr)}")
        tol = max(abs_tol, initial_rTr * rel_tol)

        # set aux fields
        if pure_neumann:
            self.calc_num_dof()
            self.recenter(self.r[0])
        self.v_cycle()

        self.p.copy_from(self.z[0])
        self.reduce(self.z[0], self.r[0])
        old_zTr = self.sum[None]

        # main loop
        iter = 0
        while max_iters == -1 or iter < max_iters:
            # self.alpha = rTr / pTAp
            self.compute_Ap()
            self.reduce(self.p, self.Ap)
            pAp = self.sum[None]
            self.alpha[None] = old_zTr / (pAp + eps)

            # self.x = self.x + self.alpha self.p
            # self.r = self.r - self.alpha self.Ap
            self.update_xr()

            # check for convergence
            self.reduce(self.r[0], self.r[0])
            rTr = self.sum[None]

            if verbose:
                print(f"iter {iter}, |residual|_2={ti.sqrt(rTr)}")

            if rTr < tol:
                break

            if pure_neumann:
                self.recenter(self.r[0])

            self.v_cycle()

            # self.beta = new_rTr / old_rTr
            self.reduce(self.z[0], self.r[0])
            new_zTr = self.sum[None]
            self.beta[None] = new_zTr / (old_zTr + eps)

            # self.p = self.z + self.beta self.p
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
            Adiag_ret = 0.0
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
                                    ti.cast(2.0, self.real) * fine_Ax[fine_coord][i]
                                )
                Adiag_ret *= 0.125
            coarse_Adiag[I] = Adiag_ret

        # Ax
        for I in ti.grouped(coarse_Ax):
            Ax_ret = ti.Vector.zero(n=3, dt=self.real)
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
                Ax_ret *= ti.cast(0.125, self.real)
            coarse_Ax[I] = Ax_ret

    @ti.kernel
    def prolongate(self, l: ti.template()):
        for I in ti.grouped(self.z[l]):
            if self.is_dof[l][I]:
                self.z[l][I] += (
                    ti.cast(2.0, self.real) * self.z[l + 1][I // 2]
                )  # 2.0 for fast convergence

    @ti.func
    def neighbor_sum(self, is_dof, Ax, x, I):
        ret = ti.cast(0.0, self.real)
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
                self.r[l + 1][I] = ret * ti.cast(0.125, self.real)
            else:
                self.r[l + 1][I] = ti.cast(0.0, self.real)

    @ti.kernel
    def reduce(self, p: ti.template(), q: ti.template()):
        self.sum[None] = ti.cast(0.0, self.real)
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
                and self.Adiag[l][I] > ti.cast(0.0, self.real)
            ):
                self.z[l][I] = (
                    self.r[l][I]
                    - self.neighbor_sum(self.is_dof[l], self.Ax[l], self.z[l], I)
                ) / self.Adiag[l][I]

    @ti.kernel
    def recenter(self, a: ti.template()):
        self.sum[None] = ti.cast(0.0, self.real)
        for I in ti.grouped(a):
            if self.is_dof[0][I]:
                self.sum[None] += a[I]
        mean = self.sum[None] / self.num_dof[None]
        for I in ti.grouped(a):
            if self.is_dof[0][I]:
                a[I] -= mean

    @ti.kernel
    def calc_num_dof(self):
        self.num_dof[None] = 0
        for I in ti.grouped(self.is_dof[0]):
            if self.is_dof[0][I]:
                self.num_dof[None] += 1

    def v_cycle(self):
        self.z[0].fill(0.0)
        for l in range(self.n_mg_levels - 1):
            for i in range(self.pre_and_post_smoothing):
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
            for i in range(self.pre_and_post_smoothing):
                self.smooth(l, 1)
                self.smooth(l, 0)