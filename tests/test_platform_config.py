"""Tests for platform-specific audio configuration."""

import os
import platform
import pytest
from voice_mode.platform_config import (
    detect_wsl,
    get_platform_config,
    get_audio_config,
    reset_audio_config,
    AudioPlatformConfig
)


def test_detect_wsl_false_on_normal_linux(monkeypatch):
    """Test WSL detection returns False on normal Linux."""
    # Mock /proc/version without WSL indicators
    def mock_exists(path):
        return path == "/proc/version"

    def mock_open(path, mode):
        import io
        return io.StringIO("Linux version 5.15.0-generic")

    monkeypatch.setattr("os.path.exists", mock_exists)
    monkeypatch.setattr("builtins.open", lambda *args, **kwargs: mock_open(*args))

    # Mock environment without WSL_DISTRO_NAME
    monkeypatch.delenv("WSL_DISTRO_NAME", raising=False)

    assert detect_wsl() is False


def test_detect_wsl_true_on_wsl(monkeypatch):
    """Test WSL detection returns True when running on WSL."""
    # Mock /proc/version with Microsoft/WSL
    def mock_exists(path):
        return path == "/proc/version"

    def mock_open(path, mode):
        import io
        return io.StringIO("Linux version 5.15.0-microsoft-standard-WSL2")

    monkeypatch.setattr("os.path.exists", mock_exists)
    monkeypatch.setattr("builtins.open", lambda *args, **kwargs: mock_open(*args))

    assert detect_wsl() is True


def test_detect_wsl_via_env_var(monkeypatch):
    """Test WSL detection via WSL_DISTRO_NAME environment variable."""
    monkeypatch.setenv("WSL_DISTRO_NAME", "Ubuntu")

    assert detect_wsl() is True


def test_get_platform_config_wsl(monkeypatch):
    """Test WSL configuration has correct settings."""
    monkeypatch.setattr("voice_mode.platform_config.detect_wsl", lambda: True)
    monkeypatch.setattr("platform.system", lambda: "Linux")

    config = get_platform_config()

    assert config.platform_name == "WSL"
    assert config.is_wsl is True
    assert config.stream_buffer_ms == 100
    assert config.stream_max_buffer == 1.0
    assert config.blocksize == 1024
    assert config.stream_chunk_size == 2048
    assert config.playback_chunk_size == 2048
    assert config.enable_backpressure is True
    assert config.max_buffer_fill_ratio == 0.6


def test_get_platform_config_macos(monkeypatch):
    """Test macOS configuration has correct settings."""
    monkeypatch.setattr("voice_mode.platform_config.detect_wsl", lambda: False)
    monkeypatch.setattr("platform.system", lambda: "Darwin")

    config = get_platform_config()

    assert config.platform_name == "macOS"
    assert config.is_wsl is False
    assert config.stream_buffer_ms == 200
    assert config.stream_max_buffer == 2.5
    assert config.blocksize == 2048
    assert config.stream_chunk_size == 8192
    assert config.playback_chunk_size == 4096
    assert config.enable_backpressure is True
    assert config.max_buffer_fill_ratio == 0.8


def test_get_platform_config_linux(monkeypatch):
    """Test Linux configuration has correct settings."""
    monkeypatch.setattr("voice_mode.platform_config.detect_wsl", lambda: False)
    monkeypatch.setattr("platform.system", lambda: "Linux")

    config = get_platform_config()

    assert config.platform_name == "Linux"
    assert config.is_wsl is False
    assert config.stream_buffer_ms == 150
    assert config.stream_max_buffer == 2.0
    assert config.blocksize == 1536
    assert config.stream_chunk_size == 4096
    assert config.playback_chunk_size == 3072
    assert config.enable_backpressure is True
    assert config.max_buffer_fill_ratio == 0.75


def test_force_platform_via_env(monkeypatch):
    """Test forcing platform via VOICEMODE_FORCE_PLATFORM."""
    monkeypatch.setenv("VOICEMODE_FORCE_PLATFORM", "wsl")
    monkeypatch.setattr("platform.system", lambda: "Darwin")  # Actually macOS

    config = get_platform_config()

    # Should use WSL config even though platform.system() returns Darwin
    assert config.platform_name == "WSL"
    assert config.is_wsl is True
    assert config.blocksize == 1024  # WSL setting, not macOS


def test_force_platform_darwin(monkeypatch):
    """Test forcing macOS platform."""
    monkeypatch.setenv("VOICEMODE_FORCE_PLATFORM", "Darwin")
    monkeypatch.setattr("platform.system", lambda: "Linux")

    config = get_platform_config()

    assert config.platform_name == "macOS"
    assert config.blocksize == 2048  # macOS setting


def test_get_audio_config_singleton():
    """Test that get_audio_config returns same instance."""
    reset_audio_config()  # Clear any cached instance

    config1 = get_audio_config()
    config2 = get_audio_config()

    assert config1 is config2  # Should be same object instance


def test_reset_audio_config():
    """Test that reset_audio_config clears the cached config."""
    config1 = get_audio_config()
    reset_audio_config()
    config2 = get_audio_config()

    # Should be different instances after reset
    assert config1 is not config2


def test_audio_platform_config_dataclass():
    """Test AudioPlatformConfig dataclass creation."""
    config = AudioPlatformConfig(
        stream_buffer_ms=150,
        stream_max_buffer=2.0,
        blocksize=1024,
        stream_chunk_size=4096,
        playback_chunk_size=2048,
        enable_backpressure=True,
        max_buffer_fill_ratio=0.75,
        platform_name="TestPlatform",
        is_wsl=False
    )

    assert config.stream_buffer_ms == 150
    assert config.stream_max_buffer == 2.0
    assert config.blocksize == 1024
    assert config.stream_chunk_size == 4096
    assert config.playback_chunk_size == 2048
    assert config.enable_backpressure is True
    assert config.max_buffer_fill_ratio == 0.75
    assert config.platform_name == "TestPlatform"
    assert config.is_wsl is False


def test_wsl_has_aggressive_backpressure(monkeypatch):
    """Test that WSL has more aggressive backpressure than other platforms."""
    monkeypatch.setattr("voice_mode.platform_config.detect_wsl", lambda: True)
    monkeypatch.setattr("platform.system", lambda: "Linux")

    wsl_config = get_platform_config()

    monkeypatch.setattr("voice_mode.platform_config.detect_wsl", lambda: False)
    monkeypatch.setattr("platform.system", lambda: "Darwin")

    reset_audio_config()
    macos_config = get_platform_config()

    # WSL should have more aggressive backpressure
    assert wsl_config.max_buffer_fill_ratio < macos_config.max_buffer_fill_ratio
    assert wsl_config.stream_max_buffer < macos_config.stream_max_buffer


def test_macos_has_larger_blocksize(monkeypatch):
    """Test that macOS has larger blocksize than WSL."""
    monkeypatch.setattr("voice_mode.platform_config.detect_wsl", lambda: True)
    monkeypatch.setattr("platform.system", lambda: "Linux")

    wsl_config = get_platform_config()

    monkeypatch.setattr("voice_mode.platform_config.detect_wsl", lambda: False)
    monkeypatch.setattr("platform.system", lambda: "Darwin")

    reset_audio_config()
    macos_config = get_platform_config()

    # macOS should have larger blocksize to prevent crackling
    assert macos_config.blocksize > wsl_config.blocksize


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
