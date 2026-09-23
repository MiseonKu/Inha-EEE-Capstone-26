# Data Dictionary

## Common feature columns

- `q*_cmd_deg`: commanded joint angle in degrees.
- `q*_encoder_deg`: measured encoder joint angle in degrees.
- `p_target_*_mm`: target position in millimeters.
- `p_vive_*_mm`: Vive-derived settled position label in millimeters.

## Metadata columns

- `source_group`: experiment group identifier.
- `source_session`: source acquisition session.
- `target_id`: planned target identifier.
- `timestamp_utc`: UTC timestamp from the source package.

## Quality columns

- `quality_flags`: source-specific row quality state.
- `tracker_valid_ratio`: valid tracker sample ratio in the label window.
- `synthetic_label`: true for the 2-link synthetic-label package.
