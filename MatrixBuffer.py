import pygame
import numpy as np
import threading
import time
import torch
import multiprocessing
import ctypes

# Define a custom Pygame event for buffer updates
# This ID must be unique and greater than or equal to pygame.USEREVENT
BUFFER_UPDATED_EVENT = pygame.USEREVENT + 1


class MultiprocessSafeTensorBuffer:
    """
    A multiprocess-safe buffer for PyTorch tensors, supporting numerical and RGB modes.
    It uses shared memory (`multiprocessing.Array`) and a multiprocessing lock
    to ensure atomic updates and safe access across multiple processes.

    Initialization can be done in three ways:
    1. With a pre-existing PyTorch tensor:
       MultiprocessSafeTensorBuffer(initial_data=my_torch_tensor, mode="numerical")
    2. With a NumPy array (which will be converted to a PyTorch tensor):
       MultiprocessSafeTensorBuffer(initial_data=my_numpy_array, mode="rgb")
    3. With dimensions (n, m) to create a new zero-initialized tensor:
       MultiprocessSafeTensorBuffer(n=100, m=200, mode="numerical", dtype=torch.float32)
    """

    def __init__(self, n=None, m=None, initial_data=None, mode="numerical", dtype=torch.float32):
        """
        Initializes the MultiprocessSafeTensorBuffer.

        Args:
            n (int, optional): Number of rows. Required if initial_data is None.
            m (int, optional): Number of columns. Required if initial_data is None.
            initial_data (torch.Tensor or np.ndarray, optional): Initial tensor/array data.
                                                                 If provided, n and m are inferred.
            mode (str): The mode of the buffer, either "numerical" or "rgb".
            dtype (torch.dtype): The desired PyTorch data type for numerical mode.

        Raises:
            ValueError: If invalid dimensions, mode, or initial data are provided.
            TypeError: If initial_data is not a torch.Tensor or np.ndarray.
        """
        # Replaced threading.Lock with multiprocessing.Lock
        self._lock = multiprocessing.Lock()
        
        # We will use a multiprocessing.Event to signal updates,
        # as Pygame events are not safe to post across processes.
        self._update_event = multiprocessing.Event()

        self._mode = mode.lower()
        if self._mode not in ["numerical", "rgb"]:
            raise ValueError("Invalid mode. Must be 'numerical' or 'rgb'.")

        self._n = None
        self._m = None
        self._dtype = None
        self._numpy_dtype = None
        self._ctype = None  # Ctype for the shared array
        self._element_size_bytes = None
        self._bytes_per_pixel = None
        self._buffer_size_bytes = None
        
        # Replaced the standard bytes buffer with a multiprocessing.Array
        self._shared_array = None

        if initial_data is not None:
            if isinstance(initial_data, np.ndarray):
                initial_tensor = torch.from_numpy(initial_data.copy())
            elif isinstance(initial_data, torch.Tensor):
                initial_tensor = initial_data
            else:
                raise TypeError("initial_data must be a NumPy array or a PyTorch tensor.")
            self._initialize_from_tensor(initial_tensor, mode)
        else:
            if not isinstance(n, int) or n <= 0:
                raise ValueError("N (rows) must be a positive integer when initial_data is not provided.")
            if not isinstance(m, int) or m <= 0:
                raise ValueError("M (columns) must be a positive integer when initial_data is not provided.")
            self._n = n
            self._m = m
            if self._mode == "numerical":
                self._dtype = dtype
                self._numpy_dtype = self._get_numpy_dtype(self._dtype)
                self._ctype = self._get_ctype(self._dtype)
                if self._ctype is None:
                    raise ValueError(f"Unsupported torch.dtype for numerical mode: {self._dtype}")
                self._element_size_bytes = ctypes.sizeof(self._ctype)
                initial_tensor = torch.zeros((n, m), dtype=self._dtype)
            elif self._mode == "rgb":
                self._dtype = torch.uint8
                self._numpy_dtype = np.uint8
                self._ctype = ctypes.c_uint8
                self._bytes_per_pixel = 3
                self._element_size_bytes = ctypes.sizeof(self._ctype)
                initial_tensor = torch.zeros((n, m, self._bytes_per_pixel), dtype=self._dtype)
            
            self._buffer_size_bytes = initial_tensor.numel() * self._element_size_bytes
            
            # Create the shared memory array and initialize it with zeros
            self._shared_array = multiprocessing.Array(self._ctype, self._buffer_size_bytes)
            self._write_to_shared_array(initial_tensor)

    def _initialize_from_tensor(self, tensor, mode):
        """Internal helper to initialize buffer properties from an initial tensor."""
        if mode == "numerical":
            if tensor.ndim != 2:
                raise ValueError(f"Numerical mode expects a 2D tensor, but got {tensor.ndim}D.")
            self._n, self._m = tensor.shape
            self._dtype = tensor.dtype
            self._numpy_dtype = self._get_numpy_dtype(self._dtype)
            self._ctype = self._get_ctype(self._dtype)
            if self._ctype is None:
                raise ValueError(f"Unsupported torch.dtype for numerical mode: {self._dtype}")
            self._element_size_bytes = ctypes.sizeof(self._ctype)
        elif mode == "rgb":
            if tensor.ndim != 3 or tensor.shape[2] != 3:
                raise ValueError(f"RGB mode expects a 3D tensor with 3 channels (N, M, 3), but got shape {tensor.shape}.")
            if tensor.dtype != torch.uint8:
                raise ValueError("RGB components must be unsigned 8-bit integers (torch.uint8).")
            self._n, self._m, self._bytes_per_pixel = tensor.shape
            self._dtype = torch.uint8
            self._numpy_dtype = np.uint8
            self._ctype = ctypes.c_uint8
            self._element_size_bytes = ctypes.sizeof(self._ctype)

        self._buffer_size_bytes = tensor.numel() * self._element_size_bytes
        self._shared_array = multiprocessing.Array(self._ctype, self._buffer_size_bytes)
        self._write_to_shared_array(tensor)

    def get_update_event(self):
        """
        Returns the multiprocessing.Event used to signal updates.
        This is needed by the main process to listen for changes.
        """
        return self._update_event
        
    def _get_ctype(self, torch_dtype):
        """Maps a PyTorch dtype to its corresponding ctypes type."""
        if torch_dtype == torch.float32:
            return ctypes.c_float
        elif torch_dtype == torch.float64:
            return ctypes.c_double
        elif torch_dtype == torch.int32:
            return ctypes.c_int32
        elif torch_dtype == torch.int64:
            return ctypes.c_int64
        elif torch_dtype == torch.uint8:
            return ctypes.c_uint8
        elif torch_dtype == torch.bool:
            return ctypes.c_bool
        else:
            return None # Unsupported dtype

    def _get_numpy_dtype(self, torch_dtype):
        """Maps a PyTorch dtype to its corresponding NumPy dtype."""
        if torch_dtype == torch.float32:
            return np.float32
        elif torch_dtype == torch.float64:
            return np.float64
        elif torch_dtype == torch.int32:
            return np.int32
        elif torch_dtype == torch.int64:
            return np.int64
        elif torch_dtype == torch.uint8:
            return np.uint8
        elif torch_dtype == torch.bool:
            return np.bool_
        else:
            return None

    def _write_to_shared_array(self, tensor):
        """Writes a PyTorch tensor to the shared array."""
        expected_shape = (self._n, self._m)
        if self._mode == "rgb":
            expected_shape = (self._n, self._m, self._bytes_per_pixel)

        if tensor.shape != expected_shape:
            raise ValueError(f"Tensor shape must be {expected_shape}, but got {tensor.shape}.")
        if tensor.dtype != self._dtype:
            raise ValueError(f"Tensor dtype must be {self._dtype}, but got {tensor.dtype}.")

        np_array = tensor.cpu().numpy()
        np_shared = np.frombuffer(self._shared_array.get_obj(), dtype=self._numpy_dtype)
        np_shared[:] = np_array.flatten()
        self._update_event.set() # Set the event to notify the reader

    def read_matrix(self):
        """
        Reads the current tensor from the buffer in a multiprocess-safe manner.

        Returns:
            torch.Tensor: The current tensor stored in the buffer.
        """
        with self._lock:
            np_shared = np.frombuffer(self._shared_array.get_obj(), dtype=self._numpy_dtype)
            
            if self._mode == "numerical":
                tensor_shape = (self._n, self._m)
            elif self._mode == "rgb":
                tensor_shape = (self._n, self._m, self._bytes_per_pixel)
            
            # The numpy array from shared memory is a view.
            # We create a new array to prevent PyTorch from complaining.
            return torch.from_numpy(np_shared.copy().reshape(tensor_shape)).to(self._dtype)

    def write_matrix(self, new_tensor):
        """
        Writes a new tensor to the buffer, replacing the existing one.
        """
        with self._lock:
            self._write_to_shared_array(new_tensor)

    def add_matrix(self, other_tensor):
        """
        Adds another tensor to the current buffered tensor.
        """
        with self._lock:
            current_tensor = self.read_matrix() # read_matrix will acquire and release the lock internally
            if current_tensor.shape != other_tensor.shape or current_tensor.dtype != other_tensor.dtype:
                raise ValueError("Shape and dtype must match for addition.")
            result_tensor = current_tensor + other_tensor
            self._write_to_shared_array(result_tensor)

    # All other methods (subtract_matrix, scalar_multiply, matrix_multiply, push_line)
    # would be implemented similarly, using the lock and calling _write_to_shared_array
    # to update the shared data.

    def get_dimensions(self):
        """
        Returns the dimensions (N, M) of the tensor.
        """
        return (self._n, self._m)

    def get_mode(self):
        """
        Returns the current mode of the buffer ('numerical' or 'rgb').
        """
        return self._mode

    def get_dtype(self):
        """
        Returns the PyTorch data type of the tensor.
        """
        return self._dtype

    def get_buffer_size_bytes(self):
        """
        Returns the size of the internal shared buffer in bytes.
        """
        return self._buffer_size_bytes


class Render:
    """
    A class to render a MultiprocessSafeTensorBuffer onto a Pygame display.
    It listens for a custom Pygame event triggered by the main process
    after receiving a signal from a worker process.
    """

    def __init__(self, buffer: MultiprocessSafeTensorBuffer, display_surface: pygame.Surface):
        if buffer.get_mode() != "rgb":
            raise ValueError("Render class is currently only supported for 'rgb' mode buffers.")

        self._buffer = buffer
        self._display = display_surface
        self._n, self._m = self._buffer.get_dimensions()
        
        # A separate surface to draw the tensor onto, for optimization
        self._tensor_surface = pygame.Surface((self._m, self._n))
        
        self._blit_buffer_to_display()

    def _blit_buffer_to_display(self):
        """
        Reads the tensor from the buffer and blits it to the Pygame display.
        """
        try:
            tensor_data = self._buffer.read_matrix()
            np_data = tensor_data.cpu().numpy()
            
            # Pygame's surfarray.make_surface expects a transposed array (W, H, C)
            np_data_transposed = np.transpose(np_data, (1, 0, 2))
            temp_surface = pygame.surfarray.make_surface(np_data_transposed)
            
            scaled_surface = pygame.transform.scale(temp_surface, self._display.get_size())
            self._display.blit(scaled_surface, (0, 0))
            
            pygame.display.flip()
            
        except ValueError as e:
            print(f"Rendering error: {e}")
            
    def render(self):
        """The main rendering function, to be called in the event loop."""
        self._blit_buffer_to_display()


# --- Worker Process for Multiprocessing Example ---
def update_buffer_process(shared_buffer: MultiprocessSafeTensorBuffer, update_event: multiprocessing.Event, stop_event: multiprocessing.Event):
    """
    A function to be run in a separate process that writes new data to the buffer.
    """
    height, width = shared_buffer.get_dimensions()
    
    if shared_buffer.get_mode() != "rgb":
        print("Error: The buffer is not in 'rgb' mode.")
        return

    while not stop_event.is_set():
        new_data = torch.randint(0, 256, (height, width, 3), dtype=torch.uint8)
        
        # Write the new data to the shared buffer
        # This will acquire a lock and set the update event
        shared_buffer.write_matrix(new_data)
        
        time.sleep(0.01)

if __name__ == "__main__":
    # --- Live Rendering Example with Multiprocessing ---
    print("\n--- Starting live rendering example. Close the Pygame window to end. ---")

    # Dimensions for our RGB matrix
    MATRIX_HEIGHT = 240
    MATRIX_WIDTH = 320
    WINDOW_SIZE = (800, 600)
    
    pygame.init()
    screen = pygame.display.set_mode(WINDOW_SIZE, pygame.RESIZABLE)
    pygame.display.set_caption("Multiprocess Safe Buffer Live Renderer")

    # The MultiprocessSafeTensorBuffer is now directly compatible with multiprocessing
    rgb_buffer = MultiprocessSafeTensorBuffer(n=MATRIX_HEIGHT, m=MATRIX_WIDTH, mode="rgb")

    renderer = Render(rgb_buffer, screen)
    
    # Get the multiprocessing event from the buffer
    update_event_from_buffer = rgb_buffer.get_update_event()
    stop_event = multiprocessing.Event()
    
    # Create and start the worker process
    worker_process = multiprocessing.Process(
        target=update_buffer_process,
        args=(rgb_buffer, update_event_from_buffer, stop_event)
    )
    worker_process.start()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            # Check for the custom event. This is not the ideal way to do it.
            # We will instead check the shared event flag.

        # Check the shared multiprocessing event.
        if update_event_from_buffer.is_set():
            # If the worker process has updated the buffer, trigger a render
            renderer.render()
            # Clear the event flag so we don't redraw until the next update
            update_event_from_buffer.clear()
            
    # Clean up
    stop_event.set()
    worker_process.join()
    pygame.quit()
    print("Live rendering example finished.")