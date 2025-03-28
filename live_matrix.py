"""
A singleton class that:

Initializes a NumPy matrix to represent the pixel colors.

Provides a global access point to the matrix for other Python files.
"""

import numpy as np

# matrix_manager.py
import numpy as np

class MatrixManager:
    _instance = None  # Stores the single instance of the class

    def __new__(cls, width=800, height=600):
        if cls._instance is None:
            cls._instance = super(MatrixManager, cls).__new__(cls)
            cls._instance.width = width
            cls._instance.height = height
            cls._instance.matrix = np.zeros((height, width, 4), dtype=np.uint8)  # RGBA color format
        return cls._instance

    def get_matrix(self):
        """Returns the pixel matrix."""
        return self.matrix

    def update_pixel(self, x, y, color):
        """Updates a single pixel in the matrix."""
        if 0 <= x < self.width and 0 <= y < self.height:
            self.matrix[y, x] = color  # NumPy uses (row, col) format

    def update_region(self, x1, y1, x2, y2, color):
        """Efficiently updates a rectangular region."""
        self.matrix[y1:y2, x1:x2] = color

    def clear_matrix(self):
        """Resets the entire matrix to black (transparent)."""
        self.matrix.fill(0)
