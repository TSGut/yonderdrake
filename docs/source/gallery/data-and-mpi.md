# Head-model data and parallel runs

The animated head demos can run under MPI. The BrainWeb animations and vessel
imaging example use the external phantom data described below.

## Running distributed

```console
mpiexec -n 4 python demos/gallery/caputo_wismer/visual_brainweb_wismer_pulse.py
mpiexec -n 4 python demos/gallery/caputo_wismer/visual_brainweb_wismer_sources.py
mpiexec -n 4 python demos/gallery/caputo_wismer/visual_skullball_wismer_pulse.py
mpiexec -n 4 python demos/gallery/caputo_wismer/visual_skullball_wismer_sources.py
mpiexec -n 4 python demos/gallery/caputo_wismer/visual_brainweb_sensor_imaging.py
mpiexec -n 4 python demos/gallery/caputo_wismer/visual_circular_sensor_imaging.py
```

The reconstruction optimizer and forward/adjoint wave solves all use the
distributed mesh. The skullball variants need no external data. The BrainWeb
examples do.

## BrainWeb data

BrainWeb data are downloaded and checksum-validated into
`demos/demo-output/.brainweb/`. For manual setup, place the normal model's
gzipped raw-byte file at

```
demos/demo-output/.brainweb/phantom_1.0mm_normal_crisp.rawb.gz
```

The phantom's publication and project page are listed in the references below.

## References

- D. L. Collins, A. P. Zijdenbos, V. Kollokian, J. G. Sled, N. J. Kabani,
  C. J. Holmes, and A. C. Evans, *Design and construction of a realistic
  digital brain phantom*, IEEE Transactions on Medical Imaging **17**(3)
  (1998), 463-468,
  [doi:10.1109/42.712135](https://doi.org/10.1109/42.712135).
- [BrainWeb: Simulated Brain Database, normal anatomical
  model](https://brainweb.bic.mni.mcgill.ca/anatomic_normal.html).
