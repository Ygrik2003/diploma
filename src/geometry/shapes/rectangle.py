from shape import *


class Rectangle(Shape):
    def __init__(self, lower_left: NDArray[dtype], upper_right: NDArray[dtype]):
        super().__init__(dim=len(lower_left))
        self.lower_left = np.asarray(lower_left, dtype=dtype)
        self.upper_right = np.asarray(upper_right, dtype=dtype)

    def get_random_points(self, density: float) -> NDArray[dtype]:
        area = np.prod(self.upper_right - self.lower_left)
        count = int(density * area)
        return np.random.uniform(
            low=self.lower_left, high=self.upper_right, size=(count, self.dim)
        )

    def get_grid_points(
        self,
        resolution: float,
        method: grid_methods,
        mask: NDArray[dtype] | None = None,
    ) -> NDArray[dtype]:
        if method != grid_methods.rectangle:
            raise ValueError("Only rectangle grid method is supported for Rectangle.")

        x_coords = np.arange(self.lower_left[0], self.upper_right[0], resolution)
        y_coords = np.arange(self.lower_left[1], self.upper_right[1], resolution)

        grid = np.array(np.meshgrid(x_coords, y_coords)).T.reshape(-1, 2)

        if mask is not None:
            mask_flattened = mask.flatten()
            return grid[mask_flattened]

        return grid

    def get_boundary_points(self, density: float) -> NDArray[dtype]:
        perimeter = 2 * (
            self.upper_right[0]
            - self.lower_left[0]
            + self.upper_right[1]
            - self.lower_left[1]
        )
        count = int(density * perimeter)

        # Generate boundary points along edges
        x_edge_top = np.linspace(self.lower_left[0], self.upper_right[0], count // 4)
        x_edge_bottom = np.linspace(self.lower_left[0], self.upper_right[0], count // 4)
        y_edge_left = np.linspace(self.lower_left[1], self.upper_right[1], count // 4)
        y_edge_right = np.linspace(self.lower_left[1], self.upper_right[1], count // 4)

        top_edge = np.column_stack(
            [x_edge_top, np.full_like(x_edge_top, self.upper_right[1])]
        )
        bottom_edge = np.column_stack(
            [x_edge_bottom, np.full_like(x_edge_bottom, self.lower_left[1])]
        )
        left_edge = np.column_stack(
            [np.full_like(y_edge_left, self.lower_left[0]), y_edge_left]
        )
        right_edge = np.column_stack(
            [np.full_like(y_edge_right, self.upper_right[0]), y_edge_right]
        )

        return np.vstack([top_edge, bottom_edge, left_edge, right_edge])

    def contains(self, points: NDArray[dtype]) -> NDArray[np.bool_]:
        return np.all(
            (points >= self.lower_left) & (points <= self.upper_right), axis=1
        )
