from kivy.app import App
from kivy.uix.widget import Widget
from kivy.graphics import Rectangle
from kivy.graphics.texture import Texture
from kivy.clock import Clock
from kivy.core.window import Window
from live_matrix import MatrixManager
from kivy.config import Config
import numpy as np

"""
TODO: Fix dimension lock
"""


class TextureCanvas(Widget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.matrix_manager = MatrixManager()
        self.texture = Texture.create(size=(self.matrix_manager.width, self.matrix_manager.height))
        Clock.schedule_interval(self.update_texture, 1 / 60)  # Update at 60 FPS

    def update_texture(self, dt):
        """Convert the matrix into a byte string and update the texture."""
        matrix = self.matrix_manager.get_matrix()

        # Convert NumPy array to a raw bytes format (RGBA)
        buf = np.flip(matrix, axis=0).astype(np.uint8).tobytes()

        # Update texture data
        self.texture.blit_buffer(buf, colorfmt='rgba', bufferfmt='ubyte')

        # Set the texture to the widget
        self.canvas.clear()
        with self.canvas:
            self.rect = self.canvas.add(Rectangle(texture=self.texture, pos=self.pos, size=self.size))

class TextureApp(App):
    def __init__(self, dimension_locked=False, **kwargs):
        super().__init__(**kwargs)
        self.dimension_locked = dimension_locked
        if dimension_locked:
            # Set window size to match matrix dimensions
            Window.size = (MatrixManager().width, MatrixManager().height)
            # Disable window resizing
            Window.resizable = False
            Config.set('graphics', 'resizable', False)

    def build(self):
        return TextureCanvas()

if __name__ == "__main__":
    # Example usage with locked dimensions
    TextureApp(dimension_locked=True).run()