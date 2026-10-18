"""Equal-weight three-point fit; retain hand-eye tilt and Z=a*BaseX+b."""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'output'
OUT.mkdir(exist_ok=True)
OLD = ROOT / 'two_point_model.json'
SOURCE = ROOT / 'camera_reference.json'


def fit():
    model = json.loads(OLD.read_text())
    source = json.loads(SOURCE.read_text())
    k = np.array(source['image_intrinsics'])
    r0 = np.array(source['camera_rotation'])
    plane = model['top_plane_base_z_equals_a_x_plus_b']
    a, b = plane['a'], plane['b']
    uv = np.array([s['uv_px'] for s in model['training_samples']] +
                  [[783.1689736237097, 586.0108971170205]])
    xy = np.array([s['measured_top_base_xyz_mm'][:2] for s in model['training_samples']] +
                  [[100.74, 1085.36]])
    rays_camera = np.c_[uv, np.ones(len(uv))] @ np.linalg.inv(k).T

    def geometry(params):
        yaw = params[3]
        rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                       [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
        rotation = rz @ r0
        center = params[:3]
        d = rays_camera @ rotation.T
        t = (a * center[0] + b - center[2]) / (d[:, 2] - a * d[:, 0])
        return center + t[:, None] * d, rotation

    initial = np.r_[model['effective_camera_center_base_mm'],
                    np.deg2rad(model['fitted_base_yaw_correction_deg'])]
    result = least_squares(lambda p: (geometry(p)[0][:, :2] - xy).ravel(), initial,
                           bounds=([-np.inf, -np.inf, 253, -.3], [np.inf, np.inf, 2000, .3]),
                           x_scale='jac', ftol=1e-13, xtol=1e-13, gtol=1e-12)
    assert result.success and np.linalg.matrix_rank(result.jac) == 4
    predicted, rotation = geometry(result.x)
    center = result.x[:3]
    ray_matrix = rotation @ np.linalg.inv(k)
    den = ray_matrix[2] - a * ray_matrix[0]
    num = a * center[0] + b - center[2]
    h = np.array([center[0] * den + num * ray_matrix[0],
                  center[1] * den + num * ray_matrix[1], den])
    h /= h[2, 2]
    pixels = np.c_[uv, np.ones(len(uv))]
    mapped = pixels @ h.T
    assert np.allclose(mapped[:, :2] / mapped[:, 2:], predicted[:, :2], atol=1e-8)
    corners = np.array([[0, 0, 1], [1279, 0, 1], [0, 1023, 1], [1279, 1023, 1]])
    domain_den = corners @ h[2]
    assert (domain_den > 0).all() or (domain_den < 0).all()
    assert np.allclose(predicted[:, 2], a * predicted[:, 0] + b, atol=1e-8)
    old_h = np.array(model['matrix'])
    old = pixels @ old_h.T
    old = old[:, :2] / old[:, 2:]
    errors = np.linalg.norm(predicted[:, :2] - xy, axis=1)
    old_errors = np.linalg.norm(old - xy, axis=1)
    assert np.sum(errors ** 2) <= np.sum(old_errors ** 2) + 1e-8
    candidates = []
    for name, pixel in [('upper', [762.1677102567216, 279.9766054109965]),
                        ('lower', uv[-1].tolist())]:
        p = old_h @ np.r_[pixel, 1]
        p = p[:2] / p[2]
        candidates.append(dict(position=name, uv_px=pixel, current_model_xy_mm=p.tolist(),
                               distance_to_supplied_xy_mm=float(np.linalg.norm(p - xy[-1]))))
    assert min(candidates, key=lambda s: s['distance_to_supplied_xy_mm'])['position'] == 'lower'
    samples = []
    for i in range(len(uv)):
        samples.append(dict(label=['left', 'right', 'lower'][i], source_image=['00214.jpg', '00214.jpg', '00218.jpg'][i],
                            uv_px=uv[i].tolist(), measured_base_xy_mm=xy[i].tolist(),
                            top_z_mm=float(a * xy[i, 0] + b),
                            z_source='measured' if i < 2 else 'inferred from the user-constrained plane; not measured',
                            fitted_top_base_xyz_mm=predicted[i].tolist(),
                            error_xy_mm=(predicted[i, :2] - xy[i]).tolist(),
                            error_norm_mm=float(errors[i]), prior_model_error_norm_mm=float(old_errors[i])))
    model.update(status='candidate_three_anchors_no_independent_validation', matrix=h.tolist(),
                 effective_camera_center_base_mm=center.tolist(), camera_to_base_rotation=rotation.tolist(),
                 fitted_base_yaw_correction_deg=float(np.rad2deg(result.x[3])),
                 source_images=['00214.jpg', '00218.jpg'], training_samples=samples,
                 fit_contract='Equal-weight least squares over 3 XY anchors for Cx,Cy,Cz and Base yaw; hand-eye tilt retained.',
                 limitations=['All three observations are training anchors; no independent validation.',
                              'Third Z inferred from Z=a*BaseX+b, not a measured additional height.',
                              'Y slope fixed to zero as requested; other workpiece thickness requires recalibration.',
                              'Geometry yaw affects XY only; robot RZ and fixed grasp Z are unchanged.'])
    model.pop('source_image', None)
    report = dict(selected_target='lower', selected_image='00218.jpg', current_model_candidates=candidates,
                  plane=plane, samples=samples, training_rmse_mm=float(np.sqrt(np.mean(errors ** 2))),
                  training_max_mm=float(errors.max()), previous_model_rmse_on_three_points_mm=float(np.sqrt(np.mean(old_errors ** 2))),
                  deployed=False, self_checks='Projection equivalence, finite image domain, fit rank and least-squares improvement passed')
    (OUT / 'tcp_xy_mapping.json').write_text(json.dumps(model, ensure_ascii=False, indent=2) + '\n')
    (OUT / 'fit-report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    fit()
