import unittest
import numpy as np

from geometry.shapes.rectangle import *


class TestRectangle(unittest.TestCase):
    def setUp(self):
        self.rect = Rectangle(
            lower_left=np.array([0.0, 0.0], dtype=dtype),
            upper_right=np.array([2.0, 3.0], dtype=dtype),
        )

    def test_initialization(self):
        rect = Rectangle(
            lower_left=np.array([0.0, 0.0], dtype=dtype),
            upper_right=np.array([1.0, 1.0], dtype=dtype),
        )
        self.assertEqual(rect.dimension, 2)

        with self.assertRaises(ValueError):
            Rectangle(
                lower_left=np.array([1.0, 1.0], dtype=dtype),
                upper_right=np.array([0.0, 0.0], dtype=dtype),
            )

    def test_contains(self):
        points = np.array([[0.5, 0.5], [1.0, 2.0], [1.9, 2.9]], dtype=dtype)
        self.assertTrue(np.all(self.rect.contains(points)))

        points = np.array([[-1.0, 1.0], [3.0, 1.0], [1.0, 4.0]], dtype=dtype)
        self.assertFalse(np.any(self.rect.contains(points)))

        edge_points = np.array([[0.0, 0.0], [2.0, 3.0], [0.0, 1.5]], dtype=dtype)
        self.assertTrue(np.all(self.rect.contains(edge_points)))

    def test_get_random_points(self):
        density = 1.0
        points = self.rect.get_random_points(density)

        self.assertGreaterEqual(points.shape[0], 5)
        self.assertLessEqual(points.shape[0], 7)
        self.assertEqual(points.shape[1], 2)
        self.assertTrue(np.all(self.rect.contains(points)))

        self.assertEqual(points.dtype, dtype)

    def test_get_grid_points(self):
        resolution = 1.0
        points = self.rect.get_grid_points(resolution, grid_methods.rectangle)
        print(points)
        expected = np.array(
            [[0.0, 0.0], [0.0, 1.0], [0.0, 2.0], [1.0, 0.0], [1.0, 1.0], [1.0, 2.0]],
            dtype=dtype,
        )
        self.assertTrue(np.allclose(np.sort(points, axis=0), np.sort(expected, axis=0)))

        mask = np.array(
            [
                True,
                False,
                True,
                False,
                True,
                False,
                True,
                False,
                True,
                False,
                True,
                False,
            ]
        )
        masked_points = self.rect.get_grid_points(1.0, grid_methods.rectangle, mask)
        self.assertEqual(masked_points.shape[0], 6)

        with self.assertRaises(ValueError):
            self.rect.get_grid_points(1.0, grid_methods.triangle)

    def test_get_boundary_points(self):
        density = 1.0
        points = self.rect.get_boundary_points(density)

        self.assertEqual(points.shape[0], 8)

        on_boundary = np.logical_or(
            np.isclose(points[:, 0], 0.0) | np.isclose(points[:, 0], 2.0),
            np.isclose(points[:, 1], 0.0) | np.isclose(points[:, 1], 3.0),
        )
        self.assertTrue(np.all(on_boundary))

        self.assertEqual(points.dtype, dtype)


if __name__ == "__main__":
    import os

    original_dir = os.getcwd()
    os.chdir(f"{original_dir}/..")
    unittest.main()
