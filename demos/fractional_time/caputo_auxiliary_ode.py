"""Monolithic auxiliary-ODE solve with a physical/memory field split."""

from firedrake import (
    Constant,
    Function,
    FunctionSpace,
    TestFunction,
    UnitIntervalMesh,
    dx,
    inner,
)

from yonderdrake import (
    AuxiliaryODE,
    BirkSong,
    CaputoDerivative,
    FractionalTimeStepper,
)


def main() -> None:
    """Run the mixed auxiliary-ODE formulation with a field split."""
    num_modes = 6
    mesh = UnitIntervalMesh(2)
    space = FunctionSpace(mesh, "CG", 1)
    u = Function(space, name="relaxation").assign(1.0)
    v = TestFunction(space)
    t = Constant(0.0)
    dt = Constant(0.05)
    residual = (inner(CaputoDerivative(u, 0.6), v) + inner(u, v)) * dx

    solver_parameters = {
        "mat_type": "aij",
        "ksp_type": "gmres",
        "ksp_rtol": 1.0e-10,
        "ksp_max_it": 100,
        "pc_type": "fieldsplit",
        "pc_fieldsplit_type": "multiplicative",
        "pc_fieldsplit_physical_fields": "0",
        "pc_fieldsplit_memory_fields": ",".join(
            str(index) for index in range(1, num_modes + 1)
        ),
        "fieldsplit_physical_ksp_type": "preonly",
        "fieldsplit_physical_pc_type": "lu",
        "fieldsplit_memory_ksp_type": "preonly",
        "fieldsplit_memory_pc_type": "lu",
    }
    stepper = FractionalTimeStepper(
        residual,
        BirkSong(num_modes),
        t,
        dt,
        u,
        formulation=AuxiliaryODE(scheme="backward_euler"),
        solver_parameters=solver_parameters,
    )

    while float(t) < 0.5 - 0.5 * float(dt):
        stepper.advance()
        t.assign(t + dt)

    print(f"u({float(t):.2f}) = {u.dat.data_ro[0]:.12f}")
    print(stepper.solver_stats())
    print(stepper.appctx["yonderdrake"])


if __name__ == "__main__":
    main()
