# P1 Model Notes

Authoritative geometry:

- `L1_effective = 274.123 mm = 0.274123 m`.
- Flexible bar length: `233 mm = 0.233 m`.
- Adapter offset: `41.123 mm = 0.041123 m`.
- `adapter_offset = L1_effective - flexible_bar_length`.

The old visible CAD dimension `145.25 mm` is not used as FK length. It is only a
non-FK visual reference if it appears in old CAD notes.

Material preset:

- `TPU95A_40x12_wall4_topbottom4_infill15_honeycomb`.
- TPU 95A, 40 mm x 12 mm cross section.
- Wall count 4, top/bottom wall count 4.
- 15 percent honeycomb infill.
- Weak-axis installation uses the thin-face bending direction.

Collision and visual simplification:

- Collision box size: `0.233 0.040 0.012`.
- Collision origin: `0.157623 0 0`.
- The collision box represents the TPU flexible bar from x = 0.041123 m to x = 0.274123 m.
- No converted STEP mesh is installed because no CAD conversion tool was available.
- The visual uses the same simple box geometry.

Inertial values are placeholders. They are included only to keep the URDF
well-formed for visualization and simple spawning. They are not validated mass
properties. The URDF includes a massless technical root link named
`p1_model_root` above `base_link`; this lets `base_link` keep inertial data for
Gazebo while avoiding the KDL root-inertia warning in `robot_state_publisher`.

The model is a rigid reference model only. It does not simulate TPU bending.
