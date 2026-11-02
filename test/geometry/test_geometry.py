# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the basic geometry classes (Interval/Rectangle/Cuboid/TimeDomain)."""

import numpy as np
import pytest

from ppsci import geometry


def _seed(seed: int = 0) -> None:
    # Sampling is stochastic: always seed locally and never rely on global state.
    np.random.seed(seed)


def test_interval_random_points_shape_and_range():
    _seed()
    geom = geometry.Interval(-1.0, 1.0)
    points = geom.random_points(16)

    assert points.shape == (16, 1)
    assert points.dtype == np.float32
    assert np.all(points >= -1.0)
    assert np.all(points <= 1.0)


def test_interval_is_inside_and_on_boundary():
    geom = geometry.Interval(-1.0, 1.0)
    x = np.array([[-2.0], [-1.0], [0.0], [1.0], [2.0]])

    np.testing.assert_array_equal(geom.is_inside(x), [False, True, True, True, False])
    np.testing.assert_array_equal(
        geom.on_boundary(x), [False, True, False, True, False]
    )


def test_interval_sdf_and_derivatives():
    geom = geometry.Interval(0.0, 1.0)
    x = np.array([[0.0], [0.5], [1.0]])

    # Negative inside, zero on the boundary.
    np.testing.assert_allclose(geom.sdf_func(x).ravel(), [0.0, -0.5, 0.0], atol=1e-6)
    np.testing.assert_allclose(
        geom.sdf_derivatives(x).ravel(), [-1.0, 0.0, 1.0], rtol=1e-3, atol=2e-3
    )

    with pytest.raises(ValueError):
        geom.sdf_func(np.zeros((3, 2)))


def test_interval_sample_interior_keys_and_sdf():
    _seed()
    geom = geometry.Interval(0.0, 1.0)
    out = geom.sample_interior(8, compute_sdf_derivatives=True)

    assert set(out) == {"x", "sdf", "sdf__x"}
    for value in out.values():
        assert value.shape == (8, 1)
        assert np.isfinite(value).all()
    assert out["x"].dtype == np.float32
    # sample_interior negates sdf_func so that weights stay positive.
    assert np.all(out["sdf"] >= -1e-5)
    np.testing.assert_allclose(out["sdf"], -geom.sdf_func(out["x"]), atol=1e-6)


def test_interval_sample_boundary_keys_and_normals():
    geom = geometry.Interval(0.0, 1.0)

    out = geom.sample_boundary(2)
    assert set(out) == {"x", "normal_x"}
    np.testing.assert_array_equal(out["x"].ravel(), [0.0, 1.0])
    np.testing.assert_array_equal(out["normal_x"].ravel(), [-1.0, 1.0])

    _seed()
    out = geom.sample_boundary(7)
    assert out["x"].shape == (7, 1)
    assert out["normal_x"].shape == (7, 1)
    assert geom.on_boundary(out["x"]).all()
    np.testing.assert_allclose(np.abs(out["normal_x"]), 1.0, atol=1e-6)


def test_rectangle_properties_and_membership():
    geom = geometry.Rectangle((0.0, 0.0), (2.0, 3.0))

    assert geom.ndim == 2
    assert geom.dim_keys == ("x", "y")
    assert geom.area == 6.0
    assert geom.perimeter == 10.0
    np.testing.assert_allclose(geom.diam, np.sqrt(13.0), rtol=1e-6)

    x = np.array(
        [
            [0.0, 0.0],  # corner
            [2.0, 3.0],  # corner
            [1.0, 1.5],  # interior
            [0.0, 1.5],  # edge
            [3.0, 1.0],  # outside
            [1.0, 4.0],  # outside
        ]
    )
    np.testing.assert_array_equal(
        geom.is_inside(x), [True, True, True, True, False, False]
    )
    np.testing.assert_array_equal(
        geom.on_boundary(x), [True, True, False, True, False, False]
    )


def test_rectangle_random_points_and_sample_interior():
    _seed()
    geom = geometry.Rectangle((0.0, 0.0), (2.0, 3.0))

    points = geom.random_points(10)
    assert points.shape == (10, 2)
    assert np.all(points[:, 0] >= 0.0) and np.all(points[:, 0] <= 2.0)
    assert np.all(points[:, 1] >= 0.0) and np.all(points[:, 1] <= 3.0)

    out = geom.sample_interior(6, compute_sdf_derivatives=True)
    assert set(out) == {"x", "y", "sdf", "sdf__x", "sdf__y"}
    for value in out.values():
        assert value.shape == (6, 1)
    assert out["x"].dtype == np.float32
    assert np.all(out["sdf"] >= -1e-5)


def test_rectangle_sdf_and_derivatives_values():
    geom = geometry.Rectangle((0.0, 0.0), (1.0, 1.0))
    x = np.array([[0.0, 0.0], [0.5, 0.5], [1.5, 1.5]])

    np.testing.assert_allclose(
        geom.sdf_func(x).ravel(),
        [0.0, -0.5, np.sqrt(0.5)],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        geom.sdf_derivatives(x),
        [[-0.5, -0.5], [0.0, 0.0], [np.sqrt(0.5), np.sqrt(0.5)]],
        rtol=1e-3,
        atol=2e-3,
    )

    with pytest.raises(ValueError):
        geom.sdf_func(np.zeros((3, 3)))


def test_rectangle_sample_boundary_keys_and_on_boundary():
    _seed()
    geom = geometry.Rectangle((0.0, 0.0), (2.0, 3.0))
    out = geom.sample_boundary(8)

    assert set(out) == {"x", "y", "normal_x", "normal_y"}
    for value in out.values():
        assert value.shape == (8, 1)

    points = np.hstack([out["x"], out["y"]])
    assert geom.on_boundary(points).all()

    normal = np.hstack([out["normal_x"], out["normal_y"]])
    # Axis-aligned unit normals: exactly one nonzero component per row.
    assert np.all(np.count_nonzero(normal, axis=1) == 1)
    np.testing.assert_allclose(np.abs(normal).sum(axis=1), 1.0, atol=1e-6)


def test_rectangle_sample_interior_with_criteria():
    _seed()
    geom = geometry.Rectangle((0.0, 0.0), (2.0, 3.0))
    out = geom.sample_interior(6, criteria=lambda x, y: x[:, 0] >= 1.0)

    assert set(out) == {"x", "y", "sdf"}
    assert out["x"].shape == (6, 1)
    assert out["y"].shape == (6, 1)
    assert out["x"].min() >= 1.0


def test_cuboid_properties_and_membership():
    geom = geometry.Cuboid((0.0, 0.0, 0.0), (1.0, 2.0, 3.0))

    assert geom.ndim == 3
    assert geom.dim_keys == ("x", "y", "z")
    assert geom.area == 22.0  # 2 * (1*2 + 2*3 + 3*1)

    x = np.array(
        [
            [0.0, 0.0, 0.0],  # corner
            [1.0, 2.0, 3.0],  # corner
            [0.5, 1.0, 1.5],  # interior
            [0.5, 1.0, 0.0],  # face
            [2.0, 1.0, 1.0],  # outside
            [0.5, 3.0, 1.0],  # outside
        ]
    )
    np.testing.assert_array_equal(
        geom.is_inside(x), [True, True, True, True, False, False]
    )
    np.testing.assert_array_equal(
        geom.on_boundary(x), [True, True, False, True, False, False]
    )


def test_cuboid_sample_interior_and_sdf_derivatives():
    _seed()
    geom = geometry.Cuboid((0.0, 0.0, 0.0), (1.0, 2.0, 3.0))
    out = geom.sample_interior(5, compute_sdf_derivatives=True)

    assert set(out) == {
        "x",
        "y",
        "z",
        "sdf",
        "sdf__x",
        "sdf__y",
        "sdf__z",
    }
    for value in out.values():
        assert value.shape == (5, 1)
    assert np.all(out["sdf"] >= -1e-5)

    np.testing.assert_allclose(
        geom.sdf_func(np.array([[0.5, 1.0, 1.5]])).ravel(), [-0.5], atol=1e-6
    )
    with pytest.raises(ValueError):
        geom.sdf_func(np.zeros((3, 2)))


def test_cuboid_sample_boundary_keys():
    _seed()
    geom = geometry.Cuboid((0.0, 0.0, 0.0), (1.0, 2.0, 3.0))
    out = geom.sample_boundary(6)

    assert set(out) == {
        "x",
        "y",
        "z",
        "normal_x",
        "normal_y",
        "normal_z",
    }
    for value in out.values():
        assert value.shape == (6, 1)

    points = np.hstack([out["x"], out["y"], out["z"]])
    assert geom.on_boundary(points).all()

    normal = np.hstack([out["normal_x"], out["normal_y"], out["normal_z"]])
    assert np.all(np.count_nonzero(normal, axis=1) == 1)
    np.testing.assert_allclose(np.abs(normal).sum(axis=1), 1.0, atol=1e-6)


def test_timedomain_construction_and_initial():
    timedomain = geometry.TimeDomain(0.0, 1.0, time_step=0.25)
    assert timedomain.num_timestamps == 4
    np.testing.assert_allclose(timedomain.timestamps, [0.0, 0.25, 0.5, 0.75], atol=1e-6)
    np.testing.assert_array_equal(
        timedomain.on_initial(np.array([0.0, 0.1])), [True, False]
    )

    timedomain = geometry.TimeDomain(0.0, 1.0, timestamps=(0.0, 0.5, 1.0))
    assert timedomain.num_timestamps == 3
    np.testing.assert_allclose(timedomain.timestamps, [0.0, 0.5, 1.0])

    with pytest.raises(AssertionError):
        geometry.TimeDomain(0.0, 1.0, time_step=0.25, timestamps=(0.0, 0.5, 1.0))


def test_timexgeometry_dim_keys_and_sdf():
    timedomain = geometry.TimeDomain(0.0, 1.0, time_step=0.25)
    geom = geometry.Rectangle((0.0, 0.0), (1.0, 1.0))
    time_geom = geometry.TimeXGeometry(timedomain, geom)

    assert time_geom.ndim == 3
    assert time_geom.dim_keys == ("t", "x", "y")
    assert hasattr(time_geom, "sdf_func")

    points = np.array([[0.1, 0.2, 0.3], [0.7, 0.9, 0.4]])
    # sdf ignores the time coordinate.
    np.testing.assert_allclose(
        time_geom.sdf_func(points), geom.sdf_func(points[:, 1:]), atol=1e-6
    )
    with pytest.raises(ValueError):
        time_geom.sdf_func(np.zeros((2, 2)))

    time_geom_1d = geometry.TimeXGeometry(
        geometry.TimeDomain(0.0, 1.0, time_step=0.25),
        geometry.Interval(0.0, 1.0),
    )
    assert time_geom_1d.ndim == 2
    assert time_geom_1d.dim_keys == ("t", "x")


def test_timexgeometry_sample_interior_and_boundary():
    _seed()
    timedomain = geometry.TimeDomain(0.0, 1.0, time_step=0.25)
    geom = geometry.Rectangle((0.0, 0.0), (1.0, 1.0))
    time_geom = geometry.TimeXGeometry(timedomain, geom)

    out = time_geom.sample_interior(5, compute_sdf_derivatives=True)
    assert set(out) == {
        "t",
        "x",
        "y",
        "sdf",
        "sdf__t",
        "sdf__x",
        "sdf__y",
    }
    for value in out.values():
        assert value.shape == (5, 1)
    assert np.all(out["sdf"] >= -1e-5)
    # Interior time samples exclude the initial time t0.
    assert np.all(np.isin(np.unique(out["t"]), timedomain.timestamps[1:]))
    # sdf ignores t, so its derivative with respect to t is zero.
    np.testing.assert_allclose(out["sdf__t"], 0.0, atol=1e-6)

    _seed(1)
    out = time_geom.sample_boundary(5)
    assert set(out) == {"t", "x", "y", "normal_x", "normal_y"}
    for value in out.values():
        assert value.shape == (5, 1)

    # on_boundary of TimeXGeometry expects the full (t, x, y) array.
    points = np.hstack([out["t"], out["x"], out["y"]])
    assert time_geom.on_boundary(points).all()
    assert np.all(out["t"] >= timedomain.t0)
    assert np.all(out["t"] <= timedomain.t1)


def test_timexgeometry_initial_points():
    _seed()
    timedomain = geometry.TimeDomain(0.0, 1.0, time_step=0.25)
    geom = geometry.Rectangle((0.0, 0.0), (1.0, 1.0))
    time_geom = geometry.TimeXGeometry(timedomain, geom)

    out = time_geom.sample_initial_interior(4)
    assert set(out) == {"t", "x", "y", "sdf"}
    for value in out.values():
        assert value.shape == (4, 1)
    np.testing.assert_allclose(out["t"], timedomain.t0)

    random_points = time_geom.random_initial_points(3)
    assert random_points.shape == (3, 3)
    np.testing.assert_allclose(random_points[:, 0], timedomain.t0)

    uniform_points = time_geom.uniform_initial_points(4)
    assert uniform_points.shape == (4, 3)
    np.testing.assert_allclose(uniform_points[:, 0], timedomain.t0)


def test_timexgeometry_on_initial():
    timedomain = geometry.TimeDomain(0.0, 1.0, time_step=0.25)
    geom = geometry.Rectangle((0.0, 0.0), (1.0, 1.0))
    time_geom = geometry.TimeXGeometry(timedomain, geom)

    np.testing.assert_array_equal(
        time_geom.on_initial(np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])),
        [True, False],
    )


if __name__ == "__main__":
    pytest.main()
