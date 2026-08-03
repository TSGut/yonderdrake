Detailed API reference
======================

This page is generated from Yonderdrake's public Python objects. The concise
support tables and method-selection guidance remain in :doc:`api`.

Time-memory operators
---------------------

.. automodule:: yonderdrake

Markers
~~~~~~~

.. autofunction:: yonderdrake.CaputoDerivative

.. autofunction:: yonderdrake.RiemannLiouvilleDerivative

.. autofunction:: yonderdrake.ExponentialMemory

.. autofunction:: yonderdrake.CaputoFabrizioOperator

Representations
~~~~~~~~~~~~~~~

.. autoclass:: yonderdrake.BirkSong
   :members:

.. autoclass:: yonderdrake.Cayley
   :members: power

.. autoclass:: yonderdrake.Diethelm2008
   :members:

.. autoclass:: yonderdrake.SumOfExponentials
   :members:

.. autoclass:: yonderdrake.Diethelm2022
   :members:

.. autoclass:: yonderdrake.YuanAgrawal
   :members:

.. autoclass:: yonderdrake.SineDiffusive
   :members:

.. autoclass:: yonderdrake.FullHistory
   :members:

Formulations and steppers
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: yonderdrake.Recurrence
   :members:

.. autoclass:: yonderdrake.AuxiliaryODE
   :members:

.. autoclass:: yonderdrake.Oscillator
   :members:

.. autofunction:: yonderdrake.FractionalTimeStepper

.. autofunction:: yonderdrake.TimeMemoryStepper

.. autoclass:: yonderdrake.ExponentialMemoryCompatibilityWarning

.. autoclass:: yonderdrake.ModeCountAdvisoryWarning

.. autoclass:: yonderdrake.StartingCorrectionAdvisoryWarning

Spatial operators
-----------------

.. autofunction:: yonderdrake.SpectralFractionalLaplacian

.. autofunction:: yonderdrake.RieszFractionalLaplacian

.. autofunction:: yonderdrake.PeriodicFractionalLaplacian

Caputo-Wismer applications
---------------------------

.. automodule:: yonderdrake.applications

.. autoclass:: yonderdrake.applications.CaputoWismerMaterial
   :members:

.. autoclass:: yonderdrake.applications.SensorArray
   :members:

.. autoclass:: yonderdrake.applications.CaputoWismerStepper
   :members:

.. autoclass:: yonderdrake.applications.CaputoWismerSource
   :members:

.. autoclass:: yonderdrake.applications.CaputoWismerArraySource
   :members:

.. autoclass:: yonderdrake.applications.CaputoWismerImpedanceBoundary
   :members:

.. autoclass:: yonderdrake.applications.CaputoWismerPML
   :members:

.. autofunction:: yonderdrake.applications.ring_sensor_locations

.. autofunction:: yonderdrake.applications.sphere_sensor_locations

.. autoclass:: yonderdrake.applications.CaputoWismerModel
   :members:

.. autoclass:: yonderdrake.applications.CaputoWismerInverseProblem
   :members:

.. autoclass:: yonderdrake.applications.CaputoWismerReconstruction
   :members:

.. autofunction:: yonderdrake.applications.reconstruct_initial_pressure

.. autofunction:: yonderdrake.applications.time_reverse_sensor_data
