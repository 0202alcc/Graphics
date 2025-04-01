# text_renderer.py
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from live_matrix import MatrixManager

class TextRenderer:
    def __init__(self, font_path, font_size):
        self.font_path = font_path
        self.font_size = font_size
        self.font = ImageFont.truetype(self.font_path, self.font_size)
        self.matrix_manager = MatrixManager()

    def render_text(self, text, start_x, start_y, color):
        # Create an image with a transparent background
        image = Image.new("RGBA", (self.matrix_manager.width, self.matrix_manager.height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)

        # Render the text onto the image
        draw.text((start_x, start_y), text, font=self.font, fill=color)

        # Convert the image to a numpy array
        image_array = np.array(image)

        # Calculate the bounding box of the text using the non-zero area of the image_array
        non_zero_indices = np.argwhere(image_array[:, :, 3] > 0)  # Find non-transparent pixels
        if non_zero_indices.size > 0:
            min_y, min_x = non_zero_indices.min(axis=0)[:2]
            max_y, max_x = non_zero_indices.max(axis=0)[:2]
            text_width = max_x - min_x + 1
            text_height = max_y - min_y + 1
        else:
            text_width, text_height = 0, 0

        end_x = min(start_x + text_width, self.matrix_manager.width)
        end_y = min(start_y + text_height, self.matrix_manager.height)

        # Update only the relevant portion of the matrix
        matrix = self.matrix_manager.get_matrix()
        matrix[start_y:end_y, start_x:end_x] = image_array[start_y:end_y, start_x:end_x]

    def get_matrix(self):
        """Returns the current state of the matrix."""
        return self.matrix_manager.get_matrix()

def main():
    # Define text and font properties
    font_path = "path/to/font.ttf"  # Update with the path to your font file
    font_size = 20
    text = "Hello, World!"
    start_x, start_y = 0, 0
    color = (255, 255, 255, 255)  # White color with full opacity

    # Create a TextRenderer instance
    text_renderer = TextRenderer(font_path, font_size)

    # Render the text
    text_renderer.render_text(text, start_x, start_y, color)

    # Access the matrix to verify the update (for debugging purposes)
    print(text_renderer.get_matrix())

if __name__ == "__main__":
    main()