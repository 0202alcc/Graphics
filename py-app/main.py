from kivy_render import TextureApp
from live_matrix import MatrixManager
from text_renderer import TextRenderer 
import threading
import time
import random

matrix = MatrixManager()

def update_matrix_colors():
    """Function to update matrix colors every second"""
    # Initialize the TextRenderer
    font_path = "/Users/aleccandidato/Projects/Graphics/py-app/fonts/ComicMono-Bold.ttf"  # Update with the path to your font file
    font_size = 20
    text_renderer = TextRenderer(font_path, font_size)

    while True:
        # Generate random colors for demonstration
        # Each color is an RGBA tuple with values 0-255
        color = (
            random.randint(0, 255),  # Red
            random.randint(0, 255),  # Green
            random.randint(0, 255),  # Blue
            255  # Alpha (fully opaque)
        )
        
        # Update the entire matrix with the new color
        matrix.update_region(0, 0, matrix.width, matrix.height, color)
        
        # Render "Hello, World!" at the top-left corner
        text_renderer.render_text("Hello, World!", 0, 0, (255, 255, 255, 255))  # White color

        # Wait for 1 second before next update
        time.sleep(1)

def main():
    # Start the matrix update thread
    update_thread = threading.Thread(target=update_matrix_colors, daemon=True)
    update_thread.start()
    
    # Run the Kivy app
    TextureApp(dimension_locked=True).run()

if __name__ == "__main__":
    main()
