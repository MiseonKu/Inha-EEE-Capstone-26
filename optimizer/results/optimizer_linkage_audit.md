# 2-link Optimizer Linkage Audit

This is discrete candidate-search smoke evidence only.

- target count: `10`
- candidate count max: `5000`
- mean best predicted optimized error: `9.2390 mm`
- median best predicted optimized error: `9.9729 mm`
- max best predicted optimized error: `13.7233 mm`
- pass@2mm: `0.0000`

## Claim Boundary

- selected `best_q1_cmd_deg`, `best_q2_cmd_deg`, `best_q3_cmd_deg` are candidate angles, not validated motor commands.
- This is not physical execution.
- This is not continuous optimization.
- This is not final control performance.
- Next step requires continuous optimization or simulation/physical validation.
