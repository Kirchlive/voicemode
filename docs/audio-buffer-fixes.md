# Audio Buffer Management Improvements

This document describes the audio buffer management improvements implemented to fix audio playback issues on WSL/ALSA and macOS.

## Problems Addressed

### 1. WSL/ALSA Buffer Overflow (Issue #1)

**Symptom**: After approximately 30 seconds of audio playback, the buffer overflows causing audio to break and become choppy. Requires server restart.

**Root Cause**:
- HTTP streaming downloads audio chunks faster than ALSA can play them back
- No backpressure mechanism to slow down downloads when buffer is full
- PortAudio/ALSA internal buffer accumulates samples until overflow
- After overflow, samples are dropped causing choppy audio

**Solution**:
- Implemented **adaptive backpressure** in PCM streaming
- Monitor buffer fill level in real-time
- Pause HTTP download when buffer exceeds 60% capacity (configurable)
- Resume download when buffer drains to safe level
- Reduced buffer size from 2.0s to 1.0s for WSL
- Smaller chunk sizes (2048 bytes) for better flow control

### 2. macOS Audio Crackling (Issue #2)

**Symptom**: Random crackling/popping sounds during audio playback on macOS.

**Root Cause**:
- `blocksize=1024` samples is too small for Core Audio
- Frequent context switches cause timing issues
- Audio callback cannot keep up with demanded rate

**Solution**:
- Increased blocksize to 2048 samples for macOS
- Larger stream buffers (2.5s max) for stability
- Increased chunk sizes (8192 bytes) for efficiency

## Implementation Details

### Platform Detection (`voice_mode/platform_config.py`)

New module that detects the runtime platform and provides optimized audio settings:

```python
from voice_mode.platform_config import get_audio_config

config = get_audio_config()
# Returns AudioPlatformConfig with platform-specific settings
```

**Platform-Specific Settings**:

| Setting | WSL | macOS | Linux |
|---------|-----|-------|-------|
| `stream_buffer_ms` | 100ms | 200ms | 150ms |
| `stream_max_buffer` | 1.0s | 2.5s | 2.0s |
| `blocksize` | 1024 | 2048 | 1536 |
| `stream_chunk_size` | 2048 | 8192 | 4096 |
| `playback_chunk_size` | 2048 | 4096 | 3072 |
| `enable_backpressure` | Yes | Yes | Yes |
| `max_buffer_fill_ratio` | 60% | 80% | 75% |

### Backpressure Algorithm (`voice_mode/streaming.py`)

The backpressure algorithm prevents buffer overflow by:

1. **Track playback position**: Calculate how many samples should have been played based on elapsed time
2. **Monitor buffer fill**: Calculate difference between samples written and samples played
3. **Apply backpressure**: If buffer exceeds threshold, pause download
4. **Calculate wait time**: Pause duration = excess samples / sample rate
5. **Resume download**: Continue when buffer drains to safe level

**Code Example**:
```python
elapsed = time.perf_counter() - playback_start_time
expected_samples = int(elapsed * SAMPLE_RATE)
buffer_fill = total_samples_written - expected_samples
max_buffer_samples = int(platform_config.stream_max_buffer * SAMPLE_RATE)

if buffer_fill > max_buffer_samples * platform_config.max_buffer_fill_ratio:
    excess_samples = buffer_fill - (max_buffer_samples * 0.4)
    wait_time = excess_samples / SAMPLE_RATE
    await asyncio.sleep(wait_time)
```

### Enhanced Metrics

Added new metrics to track buffer health:

- `buffer_overflows`: Count of buffer overflow events
- `samples_dropped`: Number of samples dropped due to overflow
- `backpressure_pauses`: Number of times backpressure was activated

These metrics are logged and can be used for debugging.

## Configuration

### Environment Variables

Force a specific platform configuration (useful for testing):
```bash
export VOICEMODE_FORCE_PLATFORM=wsl     # Force WSL settings
export VOICEMODE_FORCE_PLATFORM=Darwin  # Force macOS settings
export VOICEMODE_FORCE_PLATFORM=Linux   # Force Linux settings
```

Existing streaming configuration variables still work but may be overridden by platform settings:
```bash
export VOICEMODE_STREAM_CHUNK_SIZE=4096
export VOICEMODE_STREAM_BUFFER_MS=150
export VOICEMODE_STREAM_MAX_BUFFER=2.0
```

## Testing

### Manual Testing

**Test WSL Configuration**:
```bash
# On WSL
voicemode config set VOICEMODE_DEBUG true
# Run conversation and monitor logs for:
# - "Platform: WSL"
# - "Backpressure pauses: X"
# - No buffer overflow warnings
```

**Test macOS Configuration**:
```bash
# On macOS
voicemode config set VOICEMODE_DEBUG true
# Run conversation and listen for crackling
# Monitor logs for "Platform: macOS"
```

**Force Platform Testing**:
```bash
# Test WSL settings on any platform
export VOICEMODE_FORCE_PLATFORM=wsl
# Run tests and verify WSL configuration is used
```

### Automated Tests

Platform detection tests in `tests/test_platform_config.py`:
```bash
pytest tests/test_platform_config.py -v
```

## Performance Impact

### Expected Improvements

**WSL/ALSA**:
- ✅ No more buffer overflows after 30 seconds
- ✅ Stable audio playback for unlimited duration
- ⚠️ Occasional brief pauses when backpressure activates (typically <100ms)
- ✅ Lower latency overall (1.0s max buffer vs 2.0s)

**macOS**:
- ✅ Eliminated or greatly reduced crackling/popping
- ✅ More stable playback with larger buffers
- ℹ️ Slightly higher latency (2.5s max buffer vs 2.0s)

### Monitoring

Enable debug logging to monitor buffer health:
```bash
voicemode config set VOICEMODE_DEBUG true
```

Watch for these log messages:
- `"Backpressure: buffer X.XXs full, pausing X.XXXs"` - Normal on WSL
- `"Buffer overflow: X samples dropped"` - Should not occur with fixes
- `"Platform: [WSL|macOS|Linux]"` - Confirms platform detection

## Troubleshooting

### Audio still choppy on WSL

1. Check that WSL detection is working:
   ```bash
   python3 -c "from voice_mode.platform_config import get_audio_config; print(get_audio_config().platform_name)"
   ```
   Should output: `WSL`

2. Force WSL platform if detection fails:
   ```bash
   export VOICEMODE_FORCE_PLATFORM=wsl
   ```

3. Check PulseAudio is running:
   ```bash
   pactl info
   ```

4. Verify ALSA plugins are installed:
   ```bash
   dpkg -l | grep alsa
   ```

### Still hearing crackling on macOS

1. Try increasing buffer size:
   ```bash
   export VOICEMODE_FORCE_PLATFORM=Darwin
   export VOICEMODE_STREAM_MAX_BUFFER=3.0
   ```

2. Check for other audio applications competing for resources

3. Monitor CPU usage during playback

### Backpressure too aggressive

If you hear frequent pauses on WSL, adjust the threshold:
```python
# Edit voice_mode/platform_config.py
max_buffer_fill_ratio=0.75,  # Changed from 0.6 to 0.75 (less aggressive)
```

## Future Improvements

Potential enhancements:

1. **Adaptive buffer sizing**: Dynamically adjust buffer size based on network latency
2. **Jitter buffer**: Add jitter compensation for unstable networks
3. **Quality vs. latency tradeoff**: User-configurable presets (low-latency vs. stability)
4. **Better platform detection**: Detect specific audio backends (ALSA vs. PulseAudio vs. PipeWire)
5. **Real-time buffer visualization**: GUI showing buffer fill level

## References

- [PortAudio Documentation](http://www.portaudio.com/docs/v19-doxydocs/)
- [ALSA Buffer Configuration](https://alsa.opensrc.org/Buffer_Size_Configuration)
- [Core Audio Best Practices](https://developer.apple.com/library/archive/documentation/MusicAudio/Conceptual/CoreAudioOverview/CoreAudioEssentials/CoreAudioEssentials.html)
