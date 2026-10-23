# Stroke retiming

`retime_stroke` is a standalone JAX implementation of DATAFARM/TiPToP's timing
stage. It follows the leg-wise PyRoki geometry optimization and does not modify
PyRoki's optimization variables or train the encoder.

## Optimized quantities

The clock has 64 uniformly spaced arc-fraction knots by default. The duration
shares are `softmax(profile_rail * tanh(theta))`; cumulative shares give the
normalized arrival times. A separate `span` parameter is total duration times
the encoder's training rate. Batched Adam jointly optimizes the profile and span
from 16 duration starts for 500 iterations. `RetimeConfig` controls these values.

Score candidates on `arange(n_frames) / model_rate` using the actual trained
sampling rate. Increasing span spreads the same path over more frames. The soft
mask `clip(span - frame_index, 0, 1)` gives a differentiable effective length;
`encode(..., padding_exact=True)` avoids bias leakage from padded convolutions.
Holding frame count fixed while changing the filter's physical sampling rate
would be a different objective and is not used here.

The objective is squared latent Mahalanobis distance plus boundary, clock and
duration terms. The defaults add:

- `boundary_weight * sum((boundary_speed - target_speed)^2)` (weight 50).
- `profile_smoothness_weight * mean(((knots-2) * diff(log(time_shares)))^2)`
  (weight 0.1), discouraging abrupt changes in local speed.
- `profile_deviation_weight * var(log(time_shares))` (weight 0.5), discouraging
  large speed excursions away from a uniform arc-length clock.
- `duration_weight * duration / original_duration` (weight 2), charging for time.
- `acceleration_weight * mean(acceleration^2)` (weight 0.1) and
  `jerk_weight * mean(jerk^2)` (weight 0.01), over every robot coordinate. The
  optimizer evaluates finite differences on a fixed 50 Hz physical grid,
  independent of the encoder's training rate, masking stencils beyond the end.
  Final ranking evaluates the derivatives of the emitted positions at their
  actual timestep. The weights have units appropriate to the robot coordinates;
  adjust them for other joint units and task scales.

These are explicit engineering priors added to the learned style objective, not
learned DATAFARM parameters. They prevent style-only optimization from freely
introducing slow sections or rapid speed fluctuations. The same terms participate
in optimization and final emitted-candidate ranking; `regularization_cost` reports
their sum, with `motion_cost` reporting the acceleration/jerk portion. Tune the
weights to the application; lower style cost alone is not a
motion-quality guarantee.

Boundary speed is measured one
model-rate frame inside each requested end, over the encoder-mapped coordinates.
Positive boundary targets are soft. With the default `clock_interpolation="quintic"`,
zero requests zero clock speed and acceleration at that endpoint, so the sampled
positions themselves ease into rest. Emitted endpoint velocity is also pinned to zero.
Boundary targets are caller/configuration values, not learned gripper inputs.

The default latent target is the reference mean. An explicit `target_latent` or
JAX PRNG `key` can select a different target. A key draws one reference-Gaussian
target per stroke, shared by all starts and candidate ranking; there is no global
random state. `retime_plan` splits a supplied key across its strokes.

## Geometry and emission

The clock uses monotone quintic interpolation from normalized time to path
progress. Shared knot slopes and zero knot second derivatives make it C2;
bounded Bezier control points prevent backward motion and endpoint overshoot.
The optimizer and emitter evaluate the same clock. For comparison with the
earlier linear-clock implementation, select `clock_interpolation="linear"` and
set all five regularization weights to zero. Adam's default profile and
duration learning rates are now 0.02 and 0.005 of the allowed span range.

The clock and geometric path are represented separately. With `smoothing=0`, the
continuous path retains all original polyline corners, independent of clock-knot
count. Coincident consecutive points are removed for arc-length parameterization.
With explicit positive smoothing, SciPy fits a smoothed curve first; endpoints
are pinned and the result requires renewed geometric feasibility validation.

The optimized curve is sampled at approximately `output_dt`. The returned stroke
has uniform timestamps, actual `dt = duration/(samples-1)`, exact original endpoint
positions, velocity and acceleration arrays, chosen target and diagnostic costs.
`source_times` preserves a nonuniform original clock for baseline scoring; if
omitted, source points are assumed uniform over `original_duration`.

Motion limits are vectors in **robot coordinate order**, including joints outside
the encoder mapping. Check interval speeds and sampled velocities, pin rest-end
velocities to zero, then recompute and check accelerations. Initial clocks remain
candidate fallbacks; final selection uses emitted style, boundary and regularization cost,
rather than only the optimizer's padded score. If no candidate meets limits and
the optional validator, raise `RetimingError`.

A continuous polyline sampled into a controller trajectory is still a geometric
approximation between samples. `trajectory_validator(positions, times)` is the
hook for collision, attachment, contact and controller-specific checks; the
motion-limit checks do not establish those conditions. The function does not
execute a robot. Source and output units must match your model and robot limits.

## Gripper-delimited plans

`group_strokes` merges consecutive `MotionLeg` steps and leaves every
`GripperEvent` object unchanged. It preserves each leg's dt in the merged source
clock, rejects discontinuous endpoints (also across gripper actions), and never
silently assumes the first seven columns are a robot's arm.

`retime_plan` retimes the groups, applies the configured small speed target at
gripper-adjacent ends, and requests rest at absolute plan ends. It preserves
commands, gripper identifiers, order and stationary `dwell_seconds`. A dwell is
not an optimization variable. `plan.events` attaches the resulting absolute local
timestamps; execution remains the caller's responsibility.

The returned full motion includes dwell samples. Velocities/accelerations are
recomputed on its possibly nonuniform timestamps and validated across joins,
including gripper reversals and transitions into/out of holds. If a stitched
junction violates limits, the whole plan raises instead of returning independently
feasible but jointly invalid strokes. Adjust boundary targets, source geometry or
duration/limit settings and retry. The optional validator sees both local candidate
strokes and the final absolute local plan timeline.

## Deliberate adaptations from the pinned upstream code

- Generic model identity/rate/dimensions and explicit joint mappings.
- An exact original polyline when smoothing is disabled; clock resolution does
  not replace path corners.
- Initial feasible-clock fallbacks and ranking by the complete emitted objective.
- Recomputed acceleration after rest pinning, and full-plan junction checks.
- Explicit PRNG keys for sampled targets rather than geometry-hashed random seeds.
- C2 monotone clock interpolation, real endpoint easing and explicit clock/duration priors.

A smooth clock cannot make an exact sharp geometric corner smooth at nonzero
speed. Supply a smooth, validated path if through-motion is needed. The pick/place
example constructs the same cubic joint path through intermediate waypoints for
both modes, clamped to rest at gripper boundaries, before applying optional
learned retiming. The library still preserves caller geometry by default.

The fixed-rate timing objective and its profile/duration gradients are compared
against a PyTorch reference in tests. Adam trajectories need not be bitwise equal
between frameworks. These tests verify numerical machinery, not physical behavior.
