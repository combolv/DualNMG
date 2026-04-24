from taichi_solver import AMGPCG_3D as REF_IMPL
import taichi as ti
import taichi.math as tm
import time
import argparse
import numpy as np
import torch
from model import Model as OUR_IMPL

if __name__ != "__main__": print("Warning: importing this file may not be expected.")
parser = argparse.ArgumentParser()
parser.add_argument('-w', '--show-window', action="store_false") # Turn this option off if you do not have a display.
parser.add_argument('-m', '--model', type=str, default='ref', choices=['ref', 'original', 'dual', 'nn0', 'nn'])

args = parser.parse_args()

ti.init(arch=ti.cuda, device_memory_GB=8.0, debug=False, default_fp=ti.f64, log_level=ti.CRITICAL)
RES = 256
DX = 0.05
DT = 1.0 / 60.0

@ti.data_oriented
class HeatTransfer3D:
    def __init__(self, saving_path_prefix=None) -> None:
        if args.model == 'ref':
            self.solver = REF_IMPL([RES, RES, RES], 2)
        else:
            self.solver = OUR_IMPL([RES, RES, RES], 2, hidden_dim=8)
        
        if args.model.startswith('nn'):
            nn_para = torch.load("checkpoint_best.pt").cpu().numpy()
            if args.model == 'nn':
                self.solver.load_para_to_taichi_field(nn_para)
            else:
                self.solver.load_para_to_taichi_field(np.zeros_like(nn_para))
        self.solver.build()
        if args.model.startswith('nn'): self.solver.build_nn()

        # for obstacle
        self.sphere_center = tm.vec3(RES/2 * DX, RES/2 * DX - 20*DX, RES/2 * DX)
        self.sphere_radius = 15 * DX
        self.sphere_vel = tm.vec3(0.0, 0.0, 0.0)
        self.origin_bool = ti.field(dtype=bool, shape=[RES, RES, RES])

        # Now we should record some stats here.
        self.saving_path_prefix = saving_path_prefix

        # For now, we reverse the supporting of the total_t input.
        self.frame_idx = 0
        self.total_equations = 0

        self.build_time_list  = []
        self.solve_time_list = []
        self.cg_iter_count_list = []

    @ti.kernel
    def build_matrix_from_dof(self, is_dof: ti.template(), Adiag: ti.template(), Ax: ti.template()):
        Adiag.fill(0.0)
        Ax.fill(0.0)

        # Apply boundary conditions on the 3D grid
        for I in ti.grouped(ti.ndrange(RES, RES)):
            i, j = I[0], I[1]
            # surounding boudaries
            is_dof[i, j, 0] = False
            is_dof[i, 0, j] = False
            is_dof[0, i, j] = False
            is_dof[i, j, RES - 1] = False
            is_dof[i, RES - 1, j] = False
            is_dof[RES - 1, i, j] = False
        
        for I in ti.grouped(is_dof):
            if all(I >= 1) and all(I < RES - 1) and is_dof[I]:
                # This is a valid fluid cell in 3D
                r = DT/(DX*DX)
                val = 0.0
                for i in ti.static(range(3)):  # For 3D, we now have 3 dimensions to check
                    offset = ti.Vector.unit(3, i)
                    if (
                        all(I - offset >= 0)
                        and all(I - offset < RES)
                        and is_dof[I - offset]
                    ):
                        alpha = 5.0
                        if self.origin_bool[I] or self.origin_bool[I - offset]:
                            alpha = 0.5
                        val += alpha * r
                    if (
                        all(I + offset >= 0)
                        and all(I + offset < RES)
                        and is_dof[I + offset]
                    ):
                        alpha = 5.0
                        if self.origin_bool[I] or self.origin_bool[I + offset]:
                            alpha = 0.5
                        val += alpha * r
                        Ax[I][i] = -alpha * r
                    if not is_dof[I + offset] and all(I+offset > 0) and all (I+offset < RES - 1):
                        val += r
                        self.solver.b[I] += r * 20.0
                    if not is_dof[I - offset] and all(I-offset > 0) and all (I-offset < RES - 1):
                        val += r
                        self.solver.b[I] += r * 20.0
                Adiag[I] = 1 + val

    @ti.kernel
    def compute_dofs(self, dof_input: ti.template(), center: tm.vec3, radius: ti.f64):
        for i, j, k in dof_input:
            pos = [(i + 0.5) * DX, (j + 0.5) * DX, (k + 0.5) * DX]
            # determine if is inside the sphere
            dist = ti.sqrt(tm.dot(pos - center, pos - center))
            if (dist <= radius):
                dof_input[i, j, k] = False
            if i == 0 or i == RES - 1 or j == 0 or j == RES - 1 or k == 0 or k == RES - 1:
                dof_input[i, j, k] = False

    def build_poisson(self):
        self.solver.is_dof[0].fill(True)
        self.compute_dofs(self.solver.is_dof[0], self.sphere_center, self.sphere_radius)
        self.build_matrix_from_dof(self.solver.is_dof[0], self.solver.Adiag[0], self.solver.Ax[0])

        ti.sync()
        start_time = time.perf_counter()
        self.solver.build()
        if args.model.startswith('nn'):
            self.solver.build_nn()
        ti.sync()
        end_time = time.perf_counter()
        print("Preconditioner Building Time:", end_time - start_time, "s.")
        self.build_time_list.append(end_time - start_time)
        
    @ti.kernel
    def compute_div(self, is_dof: ti.template(), frame:int):
        for i, j, k in self.solver.b:
            if is_dof[i,j,k]:
                self.solver.b[i,j,k] = self.solver.x[i,j,k]
 
    def moving_obstacle(self, id, dt):
        if int(id / 100) % 2:
            self.sphere_vel = tm.vec3(0.0, tm.cos(id * 2 * tm.pi / 100) * 3.0, 0.0)
        else:
            self.sphere_vel = tm.vec3(0.0, 0.0, tm.cos(id * 2 * tm.pi / 100) * 3.0)
        self.sphere_center += self.sphere_vel * dt

    def solve_and_record(self, max_iter):
        if args.model == 'ref':
            ti.sync()
            start_time = time.perf_counter()
            iter_used = self.solver.solve(max_iters=max_iter, verbose=False, rel_tol=(1e-6) ** 2)
            ti.sync()
            end_time = time.perf_counter()
        else:
            ti.sync()
            start_time = time.perf_counter()
            iter_used = self.solver.solve(max_iters=max_iter, rel_tol_sqr=(1e-6) ** 2, v_cycle_name=args.model.replace('0', ''), verbose=False)
            ti.sync()
            end_time = time.perf_counter()
        self.cg_iter_count_list.append(iter_used + 1)
        print("PCG Iter used: ", iter_used, "PCG Time: ", end_time - start_time, "s.")
        self.solve_time_list.append(end_time - start_time)


    def simulate(self, time_step, id):
        self.moving_obstacle(id, time_step)
        self.solver.b.fill(0.0)
        self.compute_div(self.solver.is_dof[0], id)
        self.build_poisson()
        self.solve_and_record(-1)

    def main(self, dt, id):
        self.simulate(dt, id)
        self.frame_idx += 1


@ti.kernel
def render_velocity_field(image: ti.template(), is_dof: ti.template(), obstacles: ti.template(), den: ti.template()):
    for i, j in image:
        if is_dof[127, i, j]:
            image[RES-j-1, i] = [0.1*den[127, i, j], 0.1*den[127, i, j], 0.1*den[127, i, j]]
        else:
            image[RES - j - 1, i] = [0.0, 0.0, 100 / 256]

if __name__ == "__main__":
    image = ti.Vector.field(3, ti.f64, shape=(RES, RES))
    image.fill(0.0)
    scene = HeatTransfer3D()
    window = ti.GUI("HeatTransfer", (RES, RES), fast_gui=True, show_gui=args.show_window)

    dt = 1 / 60
    print("Start...")
    bool_input = np.load("bunny_for_heat.npy")
    scene.origin_bool.from_numpy(bool_input)
    for frame_idx in range(100):
        if not window.running: break
        scene.main(dt, frame_idx)
        render_velocity_field(image, scene.solver.is_dof[0], scene.origin_bool, scene.solver.x)
        window.set_image(image)
        window.show()
        print(frame_idx)

    avg_cg_iter = sum(scene.cg_iter_count_list[2:]) / len(scene.cg_iter_count_list[2:])
    print("Average CG Iter = ", avg_cg_iter)
    avg_solve_time = sum(scene.solve_time_list[2:]) / len(scene.solve_time_list[2:]) + sum(scene.build_time_list[2:]) / len(scene.build_time_list[2:])
    print("Average Time = ", sum(scene.solve_time_list[2:]) / len(scene.solve_time_list[2:]) + sum(scene.build_time_list[2:]) / len(scene.build_time_list[2:]), "s.")
