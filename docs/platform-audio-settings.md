# Platform-Specific Audio Settings Quick Reference

VoiceMode automatically detects your platform and applies optimized audio buffer settings to fix known issues.

## Quick Comparison: WSL vs. macOS vs. Linux

| Setting | WSL (ALSA) | macOS (Core Audio) | Linux (PulseAudio/PipeWire) |
|---------|------------|-------------------|---------------------------|
| **Main Issue** | Buffer overflow after ~30s | Random crackling/popping | Usually stable |
| **Buffer Size** | **1.0s** (reduced) | **2.5s** (increased) | 2.0s (standard) |
| **Initial Buffer** | 100ms (faster start) | 200ms (more stable) | 150ms (balanced) |
| **Blocksize** | 1024 samples | **2048 samples** (key fix) | 1536 samples |
| **HTTP Chunk Size** | **2KB** (small) | **8KB** (large) | 4KB (standard) |
| **Playback Chunk** | 2KB | 4KB | 3KB |
| **Backpressure** | **60%** (aggressive) | 80% (moderate) | 75% (balanced) |
| **Flow Control** | ✅ Critical | ✅ Enabled | ✅ Enabled |

## Platform Detection

VoiceMode automatically detects your platform:

```bash
# Check detected platform
python3 -c "from voice_mode.platform_config import get_audio_config; \
            config = get_audio_config(); \
            print(f'Platform: {config.platform_name}'); \
            print(f'Buffer: {config.stream_max_buffer}s'); \
            print(f'Blocksize: {config.blocksize}')"
```

**Expected Output:**
- WSL: `Platform: WSL`
- macOS: `Platform: macOS`
- Linux: `Platform: Linux`

## Force Platform (Override Detection)

Useful for testing or if auto-detection fails:

```bash
# Force WSL settings
export VOICEMODE_FORCE_PLATFORM=wsl

# Force macOS settings
export VOICEMODE_FORCE_PLATFORM=Darwin

# Force Linux settings
export VOICEMODE_FORCE_PLATFORM=Linux
```

## WSL-Specific Configuration

### Problem Solved
**Buffer overflow after ~30 seconds** → Audio breaks and becomes choppy, requires server restart

### Solution Applied
- **Aggressive backpressure**: Pauses HTTP download at 60% buffer fill
- **Smaller buffers**: 1.0s max (prevents overflow)
- **Small chunks**: 2KB HTTP chunks (better flow control)

### Recommended .env Settings for WSL

```bash
# Usually NOT needed - auto-detection works
# Only set if experiencing issues:

# If detection fails:
export VOICEMODE_FORCE_PLATFORM=wsl

# If backpressure pauses are too frequent (>5 per minute):
# Manually override to be less aggressive
export VOICEMODE_STREAM_MAX_BUFFER=1.5

# For debugging:
export VOICEMODE_DEBUG=true
```

### WSL Troubleshooting

**Audio still choppy after 30 seconds?**

1. Verify WSL detection:
   ```bash
   python3 -c "from voice_mode.platform_config import get_audio_config; print(get_audio_config().platform_name)"
   # Should output: WSL
   ```

2. Check PulseAudio status:
   ```bash
   pactl info
   # Should show server info
   ```

3. Install missing ALSA packages:
   ```bash
   sudo apt-get install libasound2-dev libasound2-plugins libportaudio2 portaudio19-dev pulseaudio pulseaudio-utils
   ```

4. Check for buffer overflow in logs:
   ```bash
   # Enable debug mode
   voicemode config set VOICEMODE_DEBUG true

   # Run conversation, watch for:
   # "Backpressure pauses: X" - Should be present (normal)
   # "Buffer overflow: X samples dropped" - Should NOT appear
   ```

## macOS-Specific Configuration

### Problem Solved
**Random crackling/popping sounds** during audio playback

### Solution Applied
- **Larger blocksize**: 2048 samples (reduced context switches)
- **Larger buffers**: 2.5s max (more stable playback)
- **Large chunks**: 8KB HTTP chunks (efficient transfers)

### Recommended .env Settings for macOS

```bash
# Usually NOT needed - auto-detection works
# Only set if experiencing issues:

# If detection fails:
export VOICEMODE_FORCE_PLATFORM=Darwin

# If still hearing occasional crackling:
export VOICEMODE_STREAM_MAX_BUFFER=3.0

# For debugging:
export VOICEMODE_DEBUG=true
```

### macOS Troubleshooting

**Still hearing crackling?**

1. Verify macOS detection:
   ```bash
   python3 -c "from voice_mode.platform_config import get_audio_config; print(get_audio_config().platform_name)"
   # Should output: macOS
   ```

2. Increase buffer size:
   ```bash
   export VOICEMODE_STREAM_MAX_BUFFER=3.0
   ```

3. Check for competing audio apps:
   - Close other audio applications
   - Check Activity Monitor for high CPU usage

4. Monitor buffer health:
   ```bash
   # Enable debug mode
   voicemode config set VOICEMODE_DEBUG true

   # Run conversation, watch for:
   # "Platform: macOS"
   # "Blocksize: 2048"
   ```

## Linux (Native) Configuration

### Default Settings
- Balanced configuration optimized for modern PulseAudio/PipeWire
- 2.0s buffer, 1536 blocksize, 4KB chunks
- Usually works well without modification

### Recommended .env Settings for Linux

```bash
# Usually NOT needed - defaults work well

# For low-latency systems (gaming, real-time audio):
export VOICEMODE_STREAM_BUFFER_MS=100
export VOICEMODE_STREAM_MAX_BUFFER=1.5

# For high-latency/unstable networks:
export VOICEMODE_STREAM_MAX_BUFFER=3.0

# For debugging:
export VOICEMODE_DEBUG=true
```

## Performance Metrics

After running a conversation, check the metrics:

```bash
# Enable debug mode to see metrics
voicemode config set VOICEMODE_DEBUG true

# Run a conversation, then check logs for:
```

**Good Indicators:**
```
✅ "Platform: WSL" or "Platform: macOS" - Auto-detection working
✅ "Backpressure pauses: 3" - Flow control active (normal on WSL)
✅ "Buffer overflow: 0 samples dropped" - No overflow
✅ "TTFA: 0.250s" - Fast time-to-first-audio
```

**Warning Signs:**
```
⚠️ "Buffer overflow: 500 samples dropped" - Backpressure not working
⚠️ "Backpressure pauses: 0" on WSL - Flow control may be disabled
⚠️ Platform detection incorrect - Use VOICEMODE_FORCE_PLATFORM
```

## Manual Override (Advanced)

If you need complete manual control:

```bash
# Disable automatic platform optimization (NOT recommended)
# Create custom settings by overriding all parameters:

export VOICEMODE_STREAM_BUFFER_MS=200      # Initial buffer
export VOICEMODE_STREAM_CHUNK_SIZE=8192    # HTTP download chunk
export VOICEMODE_STREAM_MAX_BUFFER=2.5     # Max buffer seconds

# Note: This won't change blocksize or backpressure behavior
# Those are set in platform_config.py and require code changes
```

## Summary

### WSL Users
✅ **No action needed** - Auto-detection fixes buffer overflow
⚠️ May see brief pauses (backpressure working) - this is normal
🔧 If issues persist, force WSL platform: `VOICEMODE_FORCE_PLATFORM=wsl`

### macOS Users
✅ **No action needed** - Auto-detection fixes crackling
📈 Slightly higher latency (~400ms more) for stability
🔧 If issues persist, increase buffer: `VOICEMODE_STREAM_MAX_BUFFER=3.0`

### Linux Users
✅ **No action needed** - Balanced defaults work well
🎯 Can tune for specific use cases (low-latency vs. stability)

## See Also

- [Detailed Audio Buffer Fixes Documentation](./audio-buffer-fixes.md)
- [Configuration Guide](../CLAUDE.md)
- [Platform Config Source Code](../voice_mode/platform_config.py)
