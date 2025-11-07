"""
Platform-specific audio configuration.

This module detects the platform and provides optimized audio settings
to prevent buffer overflow issues on WSL/ALSA and crackling on macOS.
"""

import os
import platform
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class AudioPlatformConfig:
    """Platform-specific audio configuration."""

    # Buffer settings
    stream_buffer_ms: int  # Initial buffer before playback starts
    stream_max_buffer: float  # Maximum buffer size in seconds

    # Sounddevice settings
    blocksize: int  # Audio callback block size

    # Chunk settings for streaming
    stream_chunk_size: int  # HTTP download chunk size
    playback_chunk_size: int  # Size of chunks to write to audio stream

    # Rate limiting
    enable_backpressure: bool  # Enable flow control to prevent buffer overflow
    max_buffer_fill_ratio: float  # Pause downloads when buffer exceeds this ratio

    # Platform identifier
    platform_name: str
    is_wsl: bool


def detect_wsl() -> bool:
    """Detect if running under WSL (Windows Subsystem for Linux)."""
    try:
        # Check for WSL in kernel version
        if os.path.exists("/proc/version"):
            with open("/proc/version", "r") as f:
                version = f.read().lower()
                if "microsoft" in version or "wsl" in version:
                    return True

        # Check for WSL environment variable
        if os.getenv("WSL_DISTRO_NAME"):
            return True

    except Exception as e:
        logger.debug(f"Error detecting WSL: {e}")

    return False


def get_platform_config() -> AudioPlatformConfig:
    """Get platform-specific audio configuration.

    Returns optimized settings for each platform to prevent audio issues:
    - WSL/ALSA: Smaller buffers with backpressure to prevent overflow
    - macOS: Larger blocksizes to prevent crackling
    - Linux: Balanced settings
    """
    system = platform.system()
    is_wsl = detect_wsl()

    # Allow environment variable override
    force_platform = os.getenv("VOICEMODE_FORCE_PLATFORM")
    if force_platform:
        logger.info(f"Platform forced to: {force_platform}")
        system = force_platform
        if force_platform.lower() == "wsl":
            is_wsl = True

    if is_wsl:
        # WSL with ALSA/PulseAudio
        # Problem: ALSA buffer overflows after ~30 seconds
        # Solution: Aggressive backpressure, smaller buffers, smaller chunks
        logger.info("Detected WSL - using ALSA-optimized audio settings")
        return AudioPlatformConfig(
            stream_buffer_ms=100,  # Reduced from 150ms
            stream_max_buffer=1.0,  # Reduced from 2.0s to prevent overflow
            blocksize=1024,  # Keep small for low latency on ALSA
            stream_chunk_size=2048,  # Smaller chunks (reduced from 4096)
            playback_chunk_size=2048,  # Write smaller chunks to stream
            enable_backpressure=True,  # CRITICAL: Enable flow control
            max_buffer_fill_ratio=0.6,  # Pause when 60% full (aggressive)
            platform_name="WSL",
            is_wsl=True
        )

    elif system == "Darwin":
        # macOS with Core Audio
        # Problem: Crackling/pops with small blocksizes
        # Solution: Larger blocksizes, bigger buffers for stability
        logger.info("Detected macOS - using Core Audio-optimized settings")
        return AudioPlatformConfig(
            stream_buffer_ms=200,  # Larger initial buffer
            stream_max_buffer=2.5,  # Larger max buffer for stability
            blocksize=2048,  # Increased from 1024 to reduce crackling
            stream_chunk_size=8192,  # Larger chunks for efficiency
            playback_chunk_size=4096,  # Larger playback chunks
            enable_backpressure=True,  # Enable but less aggressive
            max_buffer_fill_ratio=0.8,  # More tolerant of buffer fill
            platform_name="macOS",
            is_wsl=False
        )

    elif system == "Linux":
        # Native Linux (not WSL)
        # Balanced settings - modern PulseAudio/PipeWire is robust
        logger.info("Detected Linux - using balanced audio settings")
        return AudioPlatformConfig(
            stream_buffer_ms=150,  # Standard buffer
            stream_max_buffer=2.0,  # Standard max buffer
            blocksize=1536,  # Balanced blocksize
            stream_chunk_size=4096,  # Standard chunk size
            playback_chunk_size=3072,  # Balanced playback chunks
            enable_backpressure=True,
            max_buffer_fill_ratio=0.75,  # Moderate backpressure
            platform_name="Linux",
            is_wsl=False
        )

    else:
        # Windows or unknown platform - use conservative defaults
        logger.info(f"Detected {system} - using default audio settings")
        return AudioPlatformConfig(
            stream_buffer_ms=150,
            stream_max_buffer=2.0,
            blocksize=1536,
            stream_chunk_size=4096,
            playback_chunk_size=3072,
            enable_backpressure=True,
            max_buffer_fill_ratio=0.75,
            platform_name=system,
            is_wsl=False
        )


# Global platform config instance
_platform_config: Optional[AudioPlatformConfig] = None


def get_audio_config() -> AudioPlatformConfig:
    """Get the global platform configuration instance."""
    global _platform_config
    if _platform_config is None:
        _platform_config = get_platform_config()
    return _platform_config


def reset_audio_config():
    """Reset the platform configuration (useful for testing)."""
    global _platform_config
    _platform_config = None
