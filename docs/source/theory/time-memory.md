# Mathematics and Implementation

The supported time derivatives and their memory representations are described
together below. Power-law memory uses either stored history or diffusive modes.
Single-exponential memory is a bounded-kernel fading-memory operator with one
exact state and no quadrature spectrum. It shares the time-memory interface.

How those modes are advanced alongside the field is a separate choice, covered
in {doc}`time-stepping`.

```{include} time-derivatives.md
:heading-offset: 1
```

```{include} diffusive-representations.md
:heading-offset: 1
```

```{include} direct-time-methods.md
:heading-offset: 1
```

```{include} exponential-memory.md
:heading-offset: 1
```
