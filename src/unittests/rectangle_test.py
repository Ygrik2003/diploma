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
        # Test valid initialization
        rect = Rectangle(
            lower_left=np.array([0.0, 0.0], dtype=dtype),
            upper_right=np.array([1.0, 1.0], dtype=dtype),
        )
        self.assertEqual(rect.dimension, 2)

        # Test invalid initialization (lower_left > upper_right)
        with self.assertRaises(ValueError):
            Rectangle(
                lower_left=np.array([1.0, 1.0], dtype=dtype),
                upper_right=np.array([0.0, 0.0], dtype=dtype),
            )

    def test_contains(self):
        # Test points inside
        points = np.array([[0.5, 0.5], [1.0, 2.0], [1.9, 2.9]], dtype=dtype)
        self.assertTrue(np.all(self.rect.contains(points)))

        # Test points outside
        points = np.array([[-1.0, 1.0], [3.0, 1.0], [1.0, 4.0]], dtype=dtype)
        self.assertFalse(np.any(self.rect.contains(points)))

        # Test edge points
        edge_points = np.array([[0.0, 0.0], [2.0, 3.0], [0.0, 1.5]], dtype=dtype)
        self.assertTrue(np.all(self.rect.contains(edge_points)))

    def test_get_random_points(self):
        # Test basic functionality
        density = 1.0  # 1 point per unit area
        points = self.rect.get_random_points(density)

        # Expected points: area = 6 => ~6 points
        self.assertGreaterEqual(points.shape[0], 5)
        self.assertLessEqual(points.shape[0], 7)
        self.assertEqual(points.shape[1], 2)
        self.assertTrue(np.all(self.rect.contains(points)))

        # Test dtype preservation
        self.assertEqual(points.dtype, dtype)

    def test_get_grid_points(self):
        # Test basic rectangular grid
        resolution = 1.0
        points = self.rect.get_grid_points(resolution, grid_methods.rectangle)

        # Expected grid with 1.0 resolution in 2x3 rectangle
        expected = np.array(
            [
                [0.0, 0.0],
                [0.0, 1.0],
                [0.0, 2.0],
                [0.0, 3.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [1.0, 2.0],
                [1.0, 3.0],
                [2.0, 0.0],
                [2.0, 1.0],
                [2.0, 2.0],
                [2.0, 3.0],
            ],
            dtype=dtype,
        )

        self.assertTrue(np.allclose(np.sort(points, axis=0), np.sort(expected, axis=0)))

        # Test with mask
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

        # Test unsupported method
        with self.assertRaises(ValueError):
            self.rect.get_grid_points(1.0, grid_methods.triangle)

    def test_get_boundary_points(self):
        density = 1.0  # 1 point per unit length
        points = self.rect.get_boundary_points(density)

        # Perimeter = 2*(2+3) = 10 => ~10 points (but actual calculation uses //4 per side)
        self.assertEqual(points.shape[0], 10)  # 4 sides * (10//4) = 8-12 points

        # Verify all points are on boundaries
        on_boundary = np.logical_or(
            np.isclose(points[:, 0], 0.0) | np.isclose(points[:, 0], 2.0),
            np.isclose(points[:, 1], 0.0) | np.isclose(points[:, 1], 3.0),
        )
        self.assertTrue(np.all(on_boundary))

        # Test dtype preservation
        self.assertEqual(points.dtype, dtype)

    def test_edge_cases(self):
        # Zero-size rectangle
        zero_rect = Rectangle(
            lower_left=np.array([1.0, 1.0], dtype=dtype),
            upper_right=np.array([1.0, 1.0], dtype=dtype),
        )

        # Contains only the single point
        self.assertTrue(zero_rect.contains(np.array([[1.0, 1.0]], dtype=dtype)))
        self.assertFalse(zero_rect.contains(np.array([[1.1, 1.0]], dtype=dtype)))

        # Grid points should return empty array for zero-size rectangle
        points = zero_rect.get_grid_points(0.1, grid_methods.rectangle)
        self.assertEqual(points.shape[0], 0)

        # Boundary points should return empty array for zero-size rectangle
        points = zero_rect.get_boundary_points(1.0)
        self.assertEqual(points.shape[0], 0)

    def test_dimension_handling(self):
        # Test 1D rectangle (degenerate case)
        rect_1d = Rectangle(
            lower_left=np.array([0.0], dtype=dtype),
            upper_right=np.array([1.0], dtype=dtype),
        )

        points = rect_1d.get_random_points(10.0)
        self.assertEqual(points.shape[1], 1)
        self.assertTrue(np.all((points >= 0.0) & (points <= 1.0)))

        # Test 3D rectangle
        rect_3d = Rectangle(
            lower_left=np.array([0.0, 0.0, 0.0], dtype=dtype),
            upper_right=np.array([1.0, 1.0, 1.0], dtype=dtype),
        )
        points = rect_3d.get_random_points(1.0)
        self.assertEqual(points.shape[1], 3)


if __name__ == "__main__":
    unittest.main()
