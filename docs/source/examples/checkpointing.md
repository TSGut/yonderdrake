# Checkpoint and restart

Use Firedrake's collective checkpoint file after a successful step:

```python
stepper.advance()
t.assign(t + dt)
with CheckpointFile("restart.h5", "w", comm=mesh.comm) as checkpoint:
    stepper.save_checkpoint(checkpoint)
```

On restart, load the mesh, reconstruct the same form and stepper, then restore:

```python
with CheckpointFile("restart.h5", "r", comm=COMM_WORLD) as checkpoint:
    mesh = checkpoint.load_mesh("mesh")
    # Reconstruct V, u, F, and stepper on mesh.
    stepper.load_checkpoint(checkpoint)
```

All ranks must call both methods with the mesh communicator. The file stores
the physical field, memory, `t`, `dt`, and the configuration required to
continue. An incompatible or incomplete checkpoint raises an error without
changing the current stepper state.

Use a distinct `name` for each state in one file.

[checkpoint_restart.py](https://github.com/TSGut/yonderdrake/blob/main/demos/fractional_time/checkpoint_restart.py)
runs the complete cycle.
