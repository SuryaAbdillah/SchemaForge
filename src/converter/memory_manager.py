"""
Memory Management Module

Provides utilities for monitoring and managing memory usage during conversions.
Ensures conversions stay within specified memory limits (default 80% of available memory).
"""

import logging
import psutil
from typing import Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

# Default memory limit as percentage of total memory
DEFAULT_MEMORY_LIMIT_PERCENT = 80

# Default chunk size for processing records
DEFAULT_CHUNK_SIZE = 10000

# Minimum chunk size to avoid excessive overhead
MIN_CHUNK_SIZE = 1000

# Maximum chunk size to avoid memory issues
MAX_CHUNK_SIZE = 100000


def get_total_memory() -> int:
    """Get total system memory in bytes."""
    return psutil.virtual_memory().total


def get_available_memory() -> int:
    """Get currently available system memory in bytes."""
    return psutil.virtual_memory().available


def get_memory_usage() -> int:
    """Get current process memory usage in bytes."""
    process = psutil.Process()
    return process.memory_info().rss


def get_memory_limit(limit_percent: int = DEFAULT_MEMORY_LIMIT_PERCENT) -> int:
    """
    Calculate memory limit based on percentage of total memory.
    
    Args:
        limit_percent: Percentage of total memory to use as limit (default: 80)
    
    Returns:
        Memory limit in bytes
    """
    total = get_total_memory()
    limit = int(total * (limit_percent / 100.0))
    logger.debug(f"Memory limit set to {limit / (1024**3):.2f} GB ({limit_percent}% of {total / (1024**3):.2f} GB)")
    return limit


def get_memory_info() -> dict:
    """
    Get comprehensive memory information.
    
    Returns:
        Dictionary with memory stats in bytes and percentages
    """
    vm = psutil.virtual_memory()
    process = psutil.Process()
    
    return {
        'total': vm.total,
        'available': vm.available,
        'used': vm.used,
        'percent': vm.percent,
        'process_rss': process.memory_info().rss,
        'process_percent': process.memory_percent()
    }


def calculate_optimal_workers(memory_limit_percent: int = DEFAULT_MEMORY_LIMIT_PERCENT,
                              worker_memory_mb: int = 512) -> int:
    """
    Calculate optimal number of worker processes based on available memory.
    
    Args:
        memory_limit_percent: Percentage of total memory to use (default: 80)
        worker_memory_mb: Estimated memory per worker in MB (default: 512)
    
    Returns:
        Optimal number of workers (minimum 1, maximum 8)
    """
    available = get_available_memory()
    limit = get_memory_limit(memory_limit_percent)
    
    # Use the smaller of available or limit
    usable_memory = min(available, limit)
    
    # Calculate number of workers based on estimated memory per worker
    worker_memory_bytes = worker_memory_mb * 1024 * 1024
    max_workers = max(1, int(usable_memory / worker_memory_bytes))
    
    # Cap at reasonable maximum (8 workers) and CPU count
    cpu_count = psutil.cpu_count(logical=False) or 4
    optimal_workers = min(max_workers, cpu_count, 8)
    
    logger.info(f"Calculated optimal workers: {optimal_workers} "
                f"(available memory: {usable_memory / (1024**3):.2f} GB, "
                f"CPU cores: {cpu_count})")
    
    return optimal_workers


def calculate_chunk_size(file_path: Path, 
                        memory_limit_percent: int = DEFAULT_MEMORY_LIMIT_PERCENT,
                        estimated_record_size_kb: int = 1) -> int:
    """
    Calculate optimal chunk size for processing a file based on file size and available memory.
    
    Args:
        file_path: Path to the file to process
        memory_limit_percent: Percentage of total memory to use (default: 80)
        estimated_record_size_kb: Estimated size per record in KB (default: 1)
    
    Returns:
        Optimal chunk size (number of records per chunk)
    """
    if not file_path.exists():
        logger.warning(f"File not found: {file_path}, using default chunk size")
        return DEFAULT_CHUNK_SIZE
    
    file_size_mb = file_path.stat().st_size / (1024 * 1024)
    available_mb = get_available_memory() / (1024 * 1024)
    
    # If file is small (<50MB), process all at once
    if file_size_mb < 50:
        return MAX_CHUNK_SIZE
    
    # Calculate chunk size to use ~20% of available memory per chunk
    chunk_memory_mb = available_mb * 0.2
    chunk_size = int((chunk_memory_mb * 1024) / estimated_record_size_kb)
    
    # Clamp to min/max
    chunk_size = max(MIN_CHUNK_SIZE, min(chunk_size, MAX_CHUNK_SIZE))
    
    logger.info(f"Calculated chunk size for {file_path.name}: {chunk_size} records "
                f"(file size: {file_size_mb:.1f} MB, available memory: {available_mb:.1f} MB)")
    
    return chunk_size


def check_memory_limit(limit_percent: int = DEFAULT_MEMORY_LIMIT_PERCENT) -> Tuple[bool, float]:
    """
    Check if current memory usage is within the specified limit.
    
    Args:
        limit_percent: Percentage of total memory to use as limit (default: 80)
    
    Returns:
        Tuple of (within_limit, current_percent)
    """
    vm = psutil.virtual_memory()
    current_percent = vm.percent
    within_limit = current_percent < limit_percent
    
    if not within_limit:
        logger.warning(f"Memory usage ({current_percent:.1f}%) exceeds limit ({limit_percent}%)")
    
    return within_limit, current_percent


class MemoryMonitor:
    """
    Context manager for monitoring memory usage during operations.
    Logs warnings if memory usage exceeds specified thresholds.
    """
    
    def __init__(self, operation_name: str, 
                 limit_percent: int = DEFAULT_MEMORY_LIMIT_PERCENT,
                 check_interval_mb: int = 100):
        """
        Initialize memory monitor.
        
        Args:
            operation_name: Name of operation being monitored
            limit_percent: Memory limit percentage
            check_interval_mb: Check memory every N MB of change
        """
        self.operation_name = operation_name
        self.limit_percent = limit_percent
        self.check_interval_mb = check_interval_mb
        self.start_memory = 0
        self.peak_memory = 0
        
    def __enter__(self):
        """Start monitoring."""
        self.start_memory = get_memory_usage()
        self.peak_memory = self.start_memory
        
        mem_info = get_memory_info()
        logger.info(f"Starting {self.operation_name} - "
                   f"Process memory: {self.start_memory / (1024**2):.1f} MB, "
                   f"System memory: {mem_info['percent']:.1f}% used")
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop monitoring and log summary."""
        end_memory = get_memory_usage()
        memory_delta = end_memory - self.start_memory
        
        mem_info = get_memory_info()
        logger.info(f"Completed {self.operation_name} - "
                   f"Memory delta: {memory_delta / (1024**2):.1f} MB, "
                   f"Peak: {self.peak_memory / (1024**2):.1f} MB, "
                   f"Final system memory: {mem_info['percent']:.1f}% used")
        
        return False
    
    def check(self):
        """Check current memory usage and update peak."""
        current_memory = get_memory_usage()
        self.peak_memory = max(self.peak_memory, current_memory)
        
        # Check if we're exceeding the limit
        within_limit, current_percent = check_memory_limit(self.limit_percent)
        
        if not within_limit:
            logger.warning(f"{self.operation_name}: Memory usage high "
                          f"({current_percent:.1f}% of system memory)")
        
        return within_limit


def log_memory_stats():
    """Log current memory statistics."""
    info = get_memory_info()
    logger.info(f"Memory Stats - "
                f"Total: {info['total'] / (1024**3):.2f} GB, "
                f"Available: {info['available'] / (1024**3):.2f} GB, "
                f"Used: {info['percent']:.1f}%, "
                f"Process: {info['process_rss'] / (1024**2):.1f} MB ({info['process_percent']:.1f}%)")
