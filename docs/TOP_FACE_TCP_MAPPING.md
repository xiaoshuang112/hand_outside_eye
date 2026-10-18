# Top-face center and TCP XY mapping

The grasp center (`center_px`, `top_center_px`, `pts_top_face`) is always the intersection of the refined quadrilateral diagonals. `center_source` is `diagonal_intersection`. Image angle, legacy Z/length conversion, sorting and top-capture correspondence are unchanged.

The optional center NN is diagnostic only. Successful inference adds `center_model_center_px`, `center_model_offset_px` (NN center minus geometric center), mass, backend and inference time. Loading/inference failure records `center_model_status`/`center_model_error` and never replaces or rejects valid geometric detection. The legacy `center_model_required` setting cannot make these diagnostics mandatory.

`config/top_face/tcp_xy_mapping.json` retains `enabled` and `image_size_wh: [1280,1024]`:

- Missing or disabled: preserve legacy XY.
- Missing `model_type`, or `model_type: "affine"`: `matrix` is 2x3; `[X,Y] = matrix * [u,v,1]`.
- `model_type: "homography"`: `matrix` is 3x3; `[X,Y] = [numerator_X,numerator_Y] / denominator` from `matrix * [u,v,1]`.

Inputs are full-precision undistorted image centers; outputs are TCP XY in millimetres. Extrapolation is permitted when finite and away from a projective pole. Enabled malformed configuration fails clearly. H is normalized by its largest coefficient; singularity and denominator checks are relative to their expression terms. The denominator must be finite, nonzero and have one sign at all corners of the closed image extent `[0,1280] x [0,1024]`; runtime repeats the denominator guard for extrapolated points. Failure does not mutate the XY output references.

Configuration is cached by the existing runtime. `top_face_close`/reinitialization reloads it.

CPU checks: `tests/run_top_face_offline_checks.sh MATCHING_APP_DIR [00197.jpg 00199.jpg 00218.jpg]`. The center/homography test uses an NN test double to prove changed predictions, false results, exceptions, nonfinite output and missing/load-failed models cannot alter geometry. Actual NN source is unchanged. The conversion test supports either mapping and checks legacy Z/length/camera coordinates, image angles and ordering.
