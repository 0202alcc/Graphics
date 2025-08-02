import matplotlib.pyplot as plt
import numpy as np
import torch
import pygame
import io
from PIL import Image

# Import the necessary classes from the previous file
# Make sure matrix_buffer.py is in the same directory
from MatrixBuffer import MultiprocessSafeTensorBuffer, Render, BUFFER_UPDATED_EVENT


def generate_matplotlib_plot_as_tensor(width, height):
    """
    Generates a Matplotlib plot, saves it to an in-memory buffer, and
    returns the data as an RGB PyTorch tensor.

    Args:
        width (int): The desired width of the plot image in pixels.
        height (int): The desired height of the plot image in pixels.

    Returns:
        torch.Tensor: An RGB tensor of shape (height, width, 3) with dtype torch.uint8.
    """
    # Create a Matplotlib figure with a specific size
    # dpi (dots per inch) is used to control the final image size in pixels
    # size = (width / dpi, height / dpi)
    fig_width = width / 100
    fig_height = height / 100
    plt.style.use('dark_background')  # A dark theme for a good visual contrast
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=100)

    # Generate some data for the plot
    x = np.linspace(0, 10, 500)
    y = np.sin(x)
    y2 = np.cos(x)

    # Plot the data
    ax.plot(x, y, label='sin(x)', color='#4c72b0')
    ax.plot(x, y2, label='cos(x)', color='#55a868')

    # Customize the plot
    ax.set_title('Matplotlib Plot Rendered in Pygame', color='white')
    ax.set_xlabel('X-axis', color='white')
    ax.set_ylabel('Y-axis', color='white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white') 
    ax.spines['right'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.legend()
    fig.tight_layout()

    # Save the plot to an in-memory buffer instead of a file
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    
    # Close the figure to free up memory
    plt.close(fig)

    # Use Pillow to open the image from the buffer
    pil_image = Image.open(buf)
    
    # Ensure the image is in RGB format and convert it to a NumPy array
    np_array = np.array(pil_image.convert('RGB'))
    
    # Convert the NumPy array to a PyTorch tensor
    # The Matplotlib image is (height, width, channels)
    tensor_data = torch.from_numpy(np_array).to(torch.uint8)
    
    return tensor_data


def main():
    """
    Main function to initialize Pygame, create the plot tensor,
    and render it to the display.
    """
    # Define window dimensions and plot resolution
    WINDOW_WIDTH = 800
    WINDOW_HEIGHT = 600
    PLOT_WIDTH = 640
    PLOT_HEIGHT = 480
    
    # 1. Generate the Matplotlib plot as an RGB tensor
    print("Generating Matplotlib plot...")
    plot_tensor = generate_matplotlib_plot_as_tensor(PLOT_WIDTH, PLOT_HEIGHT)
    print("Plot tensor generated.")

    # 2. Initialize Pygame and the display
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.RESIZABLE)
    pygame.display.set_caption("Matplotlib Plot Renderer")
    screen.fill((255, 255, 255)) # Fill the screen with white background

    # 3. Create the MultiprocessSafeTensorBuffer and fill it with the plot data
    # Note: We are using MultiprocessSafeTensorBuffer, but since we are not
    # using multiple processes in this static example, a ThreadSafeTensorBuffer
    # would also work.
    rgb_buffer = MultiprocessSafeTensorBuffer(initial_data=plot_tensor, mode="rgb")

    # 4. Create the renderer to display the buffer
    # The renderer automatically draws the initial state of the buffer
    renderer = Render(rgb_buffer, screen)
    
    # 5. The main Pygame event loop
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            # Check for window resize events
            if event.type == pygame.VIDEORESIZE:
                screen = pygame.display.set_mode(event.size, pygame.RESIZABLE)
            
            # The renderer does not need to handle events here
            # because the buffer is static. If the buffer were being
            # dynamically updated, we would call `renderer.handle_event(event)`.
            
    pygame.quit()
    print("Renderer closed.")

if __name__ == "__main__":
    main()