
### Code

We provide the implementation of our multigrid solvers on the heat transfer problem using the "bunny" scene at a resolution of 256^3. The code is written using Taichi and PyTorch.

#### Dependencies

Install the required libraries:

```bash
pip install taichi torch
```

Use the following commands:
```bash
python heat_bunny.py -m ref       # Run a third-party Taichi multigrid solver
python heat_bunny.py -m original  # Run our customized GMG baseline
python heat_bunny.py -m dual      # Run the dual-channel V-cycle preconditioner
python heat_bunny.py -m nn0       # Run with our untrained model
python heat_bunny.py -m nn        # Run with our trained model
```

Note: The initial run may take some time due to Taichi kernel compilation.

During execution:

- A visualization window of the heat transfer will open.

- Logging information including CG iterations and timing.

On a workstation with an NVIDIA RTX 4080, our trained model achieves approximately 7 FPS, while other baselines typically achieve around 2 FPS.

To run in headless mode (e.g., on a server without a display), use the -w flag.
