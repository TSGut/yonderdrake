# References

The primary sources for Yonderdrake's numerical methods, software
infrastructure, and gallery examples are collected here. See
{ref}`method-map` for the method-to-source summary.

## Software infrastructure

Yonderdrake is built on Firedrake and UFL. It uses PETSc and petsc4py for
parallel linear algebra, solvers, and optimization. Its spatial operators use
Firedrake's external-operator interface.

- D. A. Ham et al., *Firedrake User Manual*, first edition (2023),
  [doi:10.25561/104839](https://doi.org/10.25561/104839).
- M. S. Alnaes, A. Logg, K. B. Ølgaard, M. E. Rognes, and G. N. Wells,
  *Unified Form Language: A domain-specific language for weak formulations of
  partial differential equations*, ACM Transactions on Mathematical Software
  40(2) (2014), Article 9,
  [doi:10.1145/2566630](https://doi.org/10.1145/2566630).
- N. Bouziani and D. A. Ham, *Escaping the abstraction: A foreign function
  interface for the Unified Form Language [UFL]*, Differentiable Programming
  Workshop at NeurIPS (2021),
  [arXiv:2111.00945](https://arxiv.org/abs/2111.00945).
- N. Bouziani, D. A. Ham, and A. Farsi, *Differentiable programming across the
  PDE and Machine Learning barrier* (2024),
  [arXiv:2409.06085](https://arxiv.org/abs/2409.06085).
- S. Balay et al., *PETSc/TAO Users Manual*, Argonne National Laboratory,
  ANL-21/39, Revision 3.25 (2026),
  [doi:10.2172/3025790](https://doi.org/10.2172/3025790).
- S. Balay, W. D. Gropp, L. Curfman McInnes, and B. F. Smith, *Efficient
  management of parallelism in object-oriented numerical software libraries*,
  in Modern Software Tools in Scientific Computing, Birkhäuser (1997),
  163-202.
- L. D. Dalcin, R. R. Paz, P. A. Kler, and A. Cosimo, *Parallel distributed
  computing using Python*, Advances in Water Resources 34(9) (2011),
  1124-1139,
  [doi:10.1016/j.advwatres.2011.04.013](https://doi.org/10.1016/j.advwatres.2011.04.013).

## Time-memory methods

- C. Lubich, *Discretized fractional calculus*, SIAM Journal on Mathematical
  Analysis 17(3) (1986), 704-719,
  [doi:10.1137/0517050](https://doi.org/10.1137/0517050).
- C. Lubich, *Convolution quadrature and discretized operational calculus. I*,
  Numerische Mathematik 52 (1988), 129-145,
  [doi:10.1007/BF01398686](https://doi.org/10.1007/BF01398686).
- C. Lubich, *Convolution quadrature and discretized operational calculus. II*,
  Numerische Mathematik 52 (1988), 413-425,
  [doi:10.1007/BF01398687](https://doi.org/10.1007/BF01398687).
- A. A. Alikhanov, *A new difference scheme for the time fractional diffusion
  equation*, Journal of Computational Physics 280 (2015), 424-438,
  [doi:10.1016/j.jcp.2014.09.031](https://doi.org/10.1016/j.jcp.2014.09.031).
- A. Schädle, M. López-Fernández, and C. Lubich, *Fast and oblivious
  convolution quadrature*, SIAM Journal on Scientific Computing 28(2) (2006),
  421-438,
  [doi:10.1137/050623139](https://doi.org/10.1137/050623139).

- M. Caputo and M. Fabrizio, *A new definition of fractional derivative
  without singular kernel*, Progress in Fractional Differentiation and
  Applications 1(2) (2015), 73-85,
  [publisher copy](https://www.naturalspublishing.com/download.asp?ArtcID=8820).
- M. D. Ortigueira and J. Tenreiro Machado, *A critical analysis of the
  Caputo-Fabrizio operator*, Communications in Nonlinear Science and
  Numerical Simulation 59 (2018), 608-611,
  [doi:10.1016/j.cnsns.2017.12.001](https://doi.org/10.1016/j.cnsns.2017.12.001).
- K. Diethelm, R. Garrappa, A. Giusti, and M. Stynes, *Why fractional
  derivatives with nonsingular kernels should not be used*, Fractional
  Calculus and Applied Analysis 23(3) (2020), 610-634,
  [doi:10.1515/fca-2020-0032](https://doi.org/10.1515/fca-2020-0032).

- Y. Lin and C. Xu, *Finite difference/spectral approximations for the
  time-fractional diffusion equation*, Journal of Computational Physics 225
  (2007), 1533-1552,
  [doi:10.1016/j.jcp.2007.02.001](https://doi.org/10.1016/j.jcp.2007.02.001).
- L. Yuan and O. P. Agrawal, *A numerical scheme for dynamic systems
  containing fractional derivatives*, Journal of Vibration and Acoustics
  124(2) (2002), 321-324,
  [doi:10.1115/1.1448322](https://doi.org/10.1115/1.1448322).
- K. Diethelm, *An investigation of some nonclassical methods for the numerical
  approximation of Caputo-type fractional derivatives*, Numerical Algorithms
  47 (2008), 361-390,
  [doi:10.1007/s11075-008-9193-8](https://doi.org/10.1007/s11075-008-9193-8).
- C. Birk and C. Song, *An improved non-classical method for the solution of
  fractional differential equations*, Computational Mechanics 46 (2010),
  721-734,
  [doi:10.1007/s00466-010-0510-4](https://doi.org/10.1007/s00466-010-0510-4).
- K. Diethelm, *Diffusive representations for the numerical evaluation of
  fractional integrals*, Proceedings of the 2023 International Conference on
  Fractional Differentiation and its Applications (2023),
  [doi:10.1109/ICFDA58234.2023.10153228](https://doi.org/10.1109/ICFDA58234.2023.10153228),
  [arXiv:2301.11931](https://arxiv.org/abs/2301.11931).
- S. Jiang, J. Zhang, Q. Zhang, and Zhimin Zhang, *Fast evaluation of the Caputo
  fractional derivative and its applications to fractional diffusion
  equations*, Communications in Computational Physics 21(3) (2017), 650-678,
  [doi:10.4208/cicp.OA-2016-0136](https://doi.org/10.4208/cicp.OA-2016-0136),
  [arXiv:1511.03453](https://arxiv.org/abs/1511.03453).
- H. Khosravian-Arab and M. Dehghan, *The sine and cosine diffusive
  representations for the Caputo fractional derivative*, Applied Numerical
  Mathematics 204 (2024), 265-290,
  [doi:10.1016/j.apnum.2024.06.017](https://doi.org/10.1016/j.apnum.2024.06.017),
  [open-access companion](https://doi.org/10.21203/rs.3.rs-2662203/v1).
- T. S. Gutleb and J. A. Carrillo, *A static memory sparse spectral method for
  time-fractional PDEs*, Journal of Computational Physics 494 (2023), 112522,
  [doi:10.1016/j.jcp.2023.112522](https://doi.org/10.1016/j.jcp.2023.112522).
- K. Diethelm, *A new diffusive representation for fractional derivatives,
  Part I: Construction, implementation and numerical examples*, in
  Fractional Differential Equations, Springer INdAM Series 50 (2023), 1-15,
  [doi:10.1007/978-981-19-7716-9_1](https://doi.org/10.1007/978-981-19-7716-9_1).
- K. Diethelm, *A new diffusive representation for fractional derivatives,
  Part II: Convergence analysis of the numerical scheme*, Mathematics 10
  (2022), 1245,
  [doi:10.3390/math10081245](https://doi.org/10.3390/math10081245).
- R. Chaudhary and K. Diethelm, *Novel variants of diffusive representation of
  fractional integrals: Construction and numerical computation*,
  IFAC-PapersOnLine 58(12) (2024), 412-417,
  [doi:10.1016/j.ifacol.2024.08.226](https://doi.org/10.1016/j.ifacol.2024.08.226).
- R. Chaudhary and K. Diethelm, *Revisiting diffusive representations for
  enhanced numerical approximation of fractional integrals*,
  IFAC-PapersOnLine 58(12) (2024), 418-423,
  [doi:10.1016/j.ifacol.2024.08.227](https://doi.org/10.1016/j.ifacol.2024.08.227).
- J. Yuan, S. Gao, G. Xiu, and B. Shi, *Equivalence of initialized
  Riemann-Liouville and Caputo derivatives*, Journal of Applied Analysis &
  Computation 10(5) (2020), 2008-2023,
  [doi:10.11948/20190317](https://doi.org/10.11948/20190317).

## Fractional spatial methods

- A. Bonito and J. E. Pasciak, *Numerical approximation of fractional powers
  of elliptic operators*, Mathematics of Computation 84 (2015), 2083-2110,
  [doi:10.1090/S0025-5718-2015-02937-8](https://doi.org/10.1090/S0025-5718-2015-02937-8).
- A. Bonito, W. Lei, and J. E. Pasciak, *On sinc quadrature approximations of
  fractional powers of regularly accretive operators*, Journal of Numerical
  Mathematics 27 (2019), 57-68,
  [doi:10.1515/jnma-2017-0116](https://doi.org/10.1515/jnma-2017-0116).
- E. Di Nezza, G. Palatucci, and E. Valdinoci, *Hitchhiker's guide to the
  fractional Sobolev spaces*, Bulletin des Sciences Mathématiques 136 (2012),
  521-573,
  [doi:10.1016/j.bulsci.2011.12.004](https://doi.org/10.1016/j.bulsci.2011.12.004).
- G. Acosta and J. P. Borthagaray, *A fractional Laplace equation: Regularity
  of solutions and finite element approximations*, SIAM Journal on Numerical
  Analysis 55 (2017), 472-495,
  [doi:10.1137/15M1033952](https://doi.org/10.1137/15M1033952).
- M. Bebendorf, *Approximation of boundary element matrices*, Numerische
  Mathematik 86 (2000), 565-589,
  [doi:10.1007/PL00005410](https://doi.org/10.1007/PL00005410).

## Fractionally attenuated acoustics

- M. G. Wismer, *Finite element analysis of broadband acoustic pulses through
  inhomogeneous media with power law attenuation*, Journal of the Acoustical
  Society of America 120(6) (2006), 3493-3502,
  doi:10.1121/1.2354032,
  [PubMed](https://pubmed.ncbi.nlm.nih.gov/17225379/).
- B. Kaltenbacher and A. Schlintl, *Fractional time stepping and adjoint based
  gradient computation in an inverse problem for a fractionally damped wave
  equation*, Journal of Computational Physics 449 (2022), 110789,
  [doi:10.1016/j.jcp.2021.110789](https://doi.org/10.1016/j.jcp.2021.110789).
- M. Kaltenbacher, B. Kaltenbacher, and I. Sim, *A modified and stable version
  of a perfectly matched layer technique for the 3-d second order wave
  equation in time domain with an application to aeroacoustics*, Journal of
  Computational Physics 235 (2013), 407-422,
  [doi:10.1016/j.jcp.2012.10.016](https://doi.org/10.1016/j.jcp.2012.10.016).
- M. J. King, T. S. Gutleb, B. E. Treeby, and B. T. Cox, *Modelling power-law
  ultrasound absorption using a time-fractional, static memory, Fourier
  pseudo-spectral method*, Journal of the Acoustical Society of America 157(3)
  (2025), 1761-1771,
  doi:10.1121/10.0035937,
  [arXiv:2408.02541](https://arxiv.org/abs/2408.02541).
- B. E. Treeby and B. T. Cox, *k-Wave: MATLAB toolbox for the simulation and
  reconstruction of photoacoustic wave fields*, Journal of Biomedical Optics
  15(2) (2010), 021314,
  doi:10.1117/1.3360308,
  [PubMed](https://pubmed.ncbi.nlm.nih.gov/20459236/).

## Gallery sources

- S. Baqer and L. Boyadjiev, *Fractional Schrödinger equation with zero and
  linear potentials*, Fractional Calculus and Applied Analysis **19**(4)
  (2016), 973-988,
  [doi:10.1515/fca-2016-0053](https://doi.org/10.1515/fca-2016-0053).
- S. Baqer and L. Boyadjiev, *On the space-time fractional Schrödinger
  equation with time independent potentials*, in Contemporary Mathematics
  **658**, American Mathematical Society (2016), 81-90,
  [doi:10.1090/conm/658/13121](https://doi.org/10.1090/conm/658/13121),
  [arXiv:1709.06198](https://arxiv.org/abs/1709.06198).
- D. Smith, J. S. Myers, C. S. Kaplan, and C. Goodman-Strauss,
  [*An aperiodic monotile*](https://cs.uwaterloo.ca/~csk/hat/),
  Combinatorial Theory **4**(1) (2024),
  [doi:10.5070/C64163843](https://doi.org/10.5070/C64163843).
