import time
import psutil
import os
from threading import Timer

class PerformanceMonitor:
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.max_memory = 0
        self.process = psutil.Process(os.getpid())
        self.timer = None

    def _record_memory(self):
        """Record the current memory usage."""
        memory_use = self.process.memory_info().rss / (1024 * 1024)  # Convert bytes to MB
        self.max_memory = max(self.max_memory, memory_use)
        if self.timer is not None:  # Check if the timer has not been stopped
            self.timer = Timer(0.1, self._record_memory)
            self.timer.start()

    def start(self):
        """Start monitoring performance."""
        self.start_time = time.time()
        self.max_memory = 0
        self._record_memory()

    def stop(self):
        """Stop monitoring performance."""
        if self.timer:
            self.timer.cancel()
            self.timer = None  # Ensure no further recordings
        self.end_time = time.time()
        self._record_memory()  # Final memory check

    def elapsed_time(self):
        """Get the elapsed time in seconds."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None

    def get_max_memory_usage(self):
        """Get the maximum memory usage in MB."""
        return self.max_memory
