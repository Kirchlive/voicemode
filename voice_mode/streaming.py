"""
Streaming audio playback for voice-mode.

This module provides progressive audio playback to reduce latency
by playing audio chunks as they arrive from the TTS service.

Platform-specific optimizations:
- WSL/ALSA: Backpressure to prevent buffer overflow
- macOS: Larger blocksizes to prevent crackling
- Linux: Balanced settings
"""

import asyncio
import io
import logging
import time
import queue
import threading
from typing import Optional, Tuple, AsyncIterator
from dataclasses import dataclass
from pathlib import Path
import numpy as np

import sounddevice as sd
from pydub import AudioSegment

from .config import (
    SAMPLE_RATE,
    logger
)
from .platform_config import get_audio_config
from .utils import get_event_logger

# Opus decoder support (optional)
try:
    import opuslib
    OPUS_AVAILABLE = True
except ImportError:
    OPUS_AVAILABLE = False
    logger.info("opuslib not available - Opus streaming will use fallback method")


@dataclass
class StreamMetrics:
    """Metrics for streaming playback performance."""
    ttfa: float = 0.0  # Time to first audio
    generation_time: float = 0.0
    playback_time: float = 0.0
    buffer_underruns: int = 0
    buffer_overflows: int = 0  # NEW: Track buffer overflow events
    samples_dropped: int = 0  # NEW: Track dropped samples
    backpressure_pauses: int = 0  # NEW: Track backpressure activations
    chunks_received: int = 0
    chunks_played: int = 0
    audio_path: Optional[str] = None  # Path to saved audio file


class AudioStreamPlayer:
    """Manages streaming audio playback with buffering."""
    
    def __init__(self, format: str, sample_rate: int = SAMPLE_RATE, channels: int = 1):
        self.format = format
        self.sample_rate = sample_rate
        self.channels = channels
        self.metrics = StreamMetrics()

        # Get platform-specific configuration
        self.platform_config = get_audio_config()
        logger.info(f"Using platform config: {self.platform_config.platform_name}")

        # Buffering (use platform-specific max buffer)
        max_queue_size = int(self.platform_config.stream_max_buffer * sample_rate)
        self.audio_queue = queue.Queue(maxsize=max_queue_size)
        self.min_buffer_samples = int((self.platform_config.stream_buffer_ms / 1000.0) * sample_rate)

        logger.debug(f"Audio queue size: {max_queue_size} samples ({self.platform_config.stream_max_buffer}s), "
                    f"min buffer: {self.min_buffer_samples} samples ({self.platform_config.stream_buffer_ms}ms)")

        # State
        self.playing = False
        self.finished_downloading = False
        self.playback_started = False
        self.start_time = time.perf_counter()

        # Partial data buffer for format-specific decoding
        self.partial_data = b''

        # Initialize decoder based on format
        self.decoder = self._get_decoder()

        # Sounddevice stream
        self.stream = None
        self._lock = threading.Lock()
        
    def _get_decoder(self):
        """Get appropriate decoder for the audio format."""
        if self.format == "opus" and OPUS_AVAILABLE:
            # Opus decoder initialization
            return opuslib.Decoder(self.sample_rate, self.channels)
        elif self.format == "pcm":
            # PCM needs no decoding
            return None
        else:
            # For MP3, AAC, etc. we'll use PyDub
            return "pydub"
    
    def _audio_callback(self, outdata, frames, time_info, status):
        """Sounddevice callback for audio playback."""
        if status:
            logger.debug(f"Sounddevice status: {status}")
            
        try:
            # Fill output buffer from queue
            for i in range(frames):
                try:
                    sample = self.audio_queue.get_nowait()
                    outdata[i] = sample
                except queue.Empty:
                    # Buffer underrun
                    outdata[i] = 0
                    if self.playing:
                        self.metrics.buffer_underruns += 1
                        
            # Track playback progress
            if self.playing:
                self.metrics.chunks_played += 1
                
        except Exception as e:
            logger.error(f"Error in audio callback: {e}")
            outdata.fill(0)
    
    async def start(self):
        """Start the audio stream with platform-optimized settings."""
        blocksize = self.platform_config.blocksize
        self.stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            callback=self._audio_callback,
            blocksize=blocksize,
            dtype='float32'
        )
        self.stream.start()
        logger.debug(f"Audio stream started (blocksize={blocksize})")
    
    async def add_chunk(self, chunk: bytes) -> bool:
        """Add an audio chunk for playback.
        
        Returns True if this was the first chunk (TTFA moment).
        """
        first_chunk = self.metrics.chunks_received == 0
        self.metrics.chunks_received += 1
        
        # Combine with any partial data
        data = self.partial_data + chunk
        
        try:
            # Decode chunk based on format
            samples = await self._decode_chunk(data)
            
            if samples is not None:
                # Successfully decoded - clear partial data
                self.partial_data = b''
                
                # Add samples to playback queue
                await self._queue_samples(samples)
                
                # Check if we should start playback
                if not self.playback_started and self.audio_queue.qsize() >= self.min_buffer_samples:
                    self.playback_started = True
                    self.playing = True
                    self.metrics.ttfa = time.perf_counter() - self.start_time
                    logger.info(f"Starting playback - TTFA: {self.metrics.ttfa:.3f}s")
                    return True
            else:
                # Partial data - save for next chunk
                self.partial_data = data
                
        except Exception as e:
            logger.error(f"Error decoding chunk: {e}")
            # Skip this chunk but try to continue
            self.partial_data = b''
            
        return first_chunk and self.playback_started
    
    async def _decode_chunk(self, data: bytes) -> Optional[np.ndarray]:
        """Decode audio chunk to samples."""
        if self.format == "pcm":
            # PCM is raw samples - just convert
            if len(data) % 2 != 0:
                # Incomplete sample - save for next chunk
                return None
            samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            return samples
            
        elif self.format == "opus" and isinstance(self.decoder, opuslib.Decoder):
            # Opus decoding
            try:
                # Opus decoder needs complete frames
                pcm = self.decoder.decode(data, frame_size=960)
                samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
                return samples
            except Exception:
                # Incomplete frame - wait for more data
                return None
                
        elif self.decoder == "pydub":
            # Use PyDub for MP3, AAC, etc.
            # This is tricky because we need complete frames
            try:
                # Try to decode what we have
                audio = AudioSegment.from_file(io.BytesIO(data), format=self.format)
                samples = np.array(audio.get_array_of_samples()).astype(np.float32) / 32768.0
                return samples
            except Exception:
                # Need more data for a complete frame
                return None
                
        return None
    
    async def _queue_samples(self, samples: np.ndarray):
        """Add samples to the playback queue with overflow tracking."""
        for sample in samples:
            try:
                self.audio_queue.put_nowait(sample)
            except queue.Full:
                # Buffer overflow - drop oldest samples
                self.metrics.buffer_overflows += 1
                try:
                    self.audio_queue.get_nowait()
                    self.audio_queue.put_nowait(sample)
                    self.metrics.samples_dropped += 1
                except queue.Empty:
                    pass

        # Log warning if we're dropping samples
        if self.metrics.samples_dropped > 0 and self.metrics.samples_dropped % 1000 == 0:
            logger.warning(f"Buffer overflow: {self.metrics.samples_dropped} samples dropped, "
                         f"{self.metrics.buffer_overflows} overflow events")
    
    async def finish(self):
        """Signal that downloading is complete."""
        self.finished_downloading = True
        self.metrics.generation_time = time.perf_counter() - self.start_time
        
        # Process any remaining partial data
        if self.partial_data:
            # For formats like MP3, we might have a complete frame now
            samples = await self._decode_chunk(self.partial_data)
            if samples is not None:
                await self._queue_samples(samples)
        
        # Wait for playback to complete
        while not self.audio_queue.empty() or self.playing:
            await asyncio.sleep(0.1)
            
        self.metrics.playback_time = time.perf_counter() - self.start_time
        
    async def stop(self):
        """Stop playback and cleanup."""
        self.playing = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
        logger.debug("Audio stream stopped")


async def stream_pcm_audio(
    text: str,
    openai_client,
    request_params: dict,
    debug: bool = False,
    save_audio: bool = False,
    audio_dir: Optional[Path] = None,
    conversation_id: Optional[str] = None
) -> Tuple[bool, StreamMetrics]:
    """Stream PCM audio with backpressure for stable playback.

    Uses platform-optimized settings to prevent buffer overflow on WSL/ALSA
    and crackling on macOS.
    """
    metrics = StreamMetrics()
    platform_config = get_audio_config()
    start_time = time.perf_counter()
    stream = None
    first_chunk_time = None
    save_buffer = io.BytesIO() if save_audio else None

    # Backpressure control
    pending_buffer = []  # Buffer for flow control
    total_samples_written = 0
    playback_start_time = None

    try:
        # Setup sounddevice stream for PCM playback with platform settings
        stream = sd.OutputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype='int16',
            blocksize=platform_config.blocksize
        )
        stream.start()

        logger.debug(f"PCM stream started with blocksize={platform_config.blocksize}, "
                    f"backpressure={'enabled' if platform_config.enable_backpressure else 'disabled'}")
        
        # Log TTS playback start
        event_logger = get_event_logger()
        if event_logger:
            event_logger.log_event(event_logger.TTS_PLAYBACK_START)

        logger.info(f"Starting PCM streaming (platform: {platform_config.platform_name})")

        # Use the streaming response API
        async with openai_client.audio.speech.with_streaming_response.create(
            **request_params
        ) as response:
            chunk_count = 0
            bytes_received = 0

            # Stream chunks as they arrive
            async for chunk in response.iter_bytes(chunk_size=platform_config.stream_chunk_size):
                if chunk:
                    # Track first chunk received
                    if first_chunk_time is None:
                        first_chunk_time = time.perf_counter()
                        chunk_receive_time = first_chunk_time - start_time
                        logger.info(f"First audio chunk received after {chunk_receive_time:.3f}s")

                        # Log TTS first audio event
                        if event_logger:
                            event_logger.log_event(event_logger.TTS_FIRST_AUDIO)

                    # Convert bytes to numpy array for sounddevice
                    audio_array = np.frombuffer(chunk, dtype=np.int16)

                    # Save chunk if enabled
                    if save_buffer:
                        save_buffer.write(chunk)

                    chunk_count += 1
                    bytes_received += len(chunk)
                    metrics.chunks_received = chunk_count

                    # BACKPRESSURE: Check buffer status before writing
                    if platform_config.enable_backpressure:
                        # Calculate expected playback position
                        if playback_start_time is None:
                            playback_start_time = time.perf_counter()

                        elapsed = time.perf_counter() - playback_start_time
                        expected_samples = int(elapsed * SAMPLE_RATE)
                        buffer_fill = total_samples_written - expected_samples
                        max_buffer_samples = int(platform_config.stream_max_buffer * SAMPLE_RATE)

                        # If buffer is too full, pause to let playback catch up
                        if buffer_fill > max_buffer_samples * platform_config.max_buffer_fill_ratio:
                            metrics.backpressure_pauses += 1
                            # Calculate how long to wait
                            excess_samples = buffer_fill - (max_buffer_samples * 0.4)  # Target 40% fill
                            wait_time = excess_samples / SAMPLE_RATE

                            if debug:
                                logger.debug(f"Backpressure: buffer {buffer_fill/SAMPLE_RATE:.2f}s full, "
                                           f"pausing {wait_time:.3f}s")

                            await asyncio.sleep(wait_time)

                    # Write audio in platform-optimized chunks
                    playback_chunk_size = platform_config.playback_chunk_size
                    for i in range(0, len(audio_array), playback_chunk_size):
                        chunk_slice = audio_array[i:i + playback_chunk_size]
                        stream.write(chunk_slice)
                        total_samples_written += len(chunk_slice)
                        metrics.chunks_played += 1

                    if debug and chunk_count % 10 == 0:
                        logger.debug(f"Streamed {chunk_count} chunks, {bytes_received} bytes, "
                                   f"{metrics.backpressure_pauses} pauses")
        
        # Wait for playback to finish
        if total_samples_written > 0:
            remaining_time = total_samples_written / SAMPLE_RATE
            if playback_start_time:
                elapsed = time.perf_counter() - playback_start_time
                remaining_time = max(0, remaining_time - elapsed)

            if remaining_time > 0:
                logger.debug(f"Waiting {remaining_time:.2f}s for playback to complete")
                await asyncio.sleep(remaining_time)

        stream.stop()

        # Log TTS playback end
        if event_logger:
            event_logger.log_event(event_logger.TTS_PLAYBACK_END)

        end_time = time.perf_counter()
        metrics.generation_time = first_chunk_time - start_time if first_chunk_time else 0
        metrics.playback_time = end_time - start_time
        metrics.ttfa = first_chunk_time - start_time if first_chunk_time else 0

        logger.info(f"PCM streaming complete - TTFA: {metrics.ttfa:.3f}s, "
                   f"Total: {metrics.playback_time:.3f}s, "
                   f"Chunks: {metrics.chunks_received}, "
                   f"Backpressure pauses: {metrics.backpressure_pauses}")
        
        # Save audio if enabled
        if save_audio and save_buffer and audio_dir:
            try:
                from .core import save_debug_file
                save_buffer.seek(0)
                audio_data = save_buffer.read()
                # PCM format needs special handling - save as WAV
                if audio_data:
                    # For PCM, we need to save as WAV with proper headers
                    import wave
                    import tempfile
                    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_wav:
                        with wave.open(tmp_wav.name, 'wb') as wav_file:
                            wav_file.setnchannels(1)
                            wav_file.setsampwidth(2)  # 16-bit
                            wav_file.setframerate(SAMPLE_RATE)
                            wav_file.writeframes(audio_data)
                        # Read back the WAV file
                        with open(tmp_wav.name, 'rb') as f:
                            wav_data = f.read()
                        import os
                        os.unlink(tmp_wav.name)
                        audio_path = save_debug_file(wav_data, "tts", "wav", audio_dir, True, conversation_id)
                        if audio_path:
                            logger.info(f"TTS audio saved to: {audio_path}")
                            # Store audio path in metrics for the caller
                            metrics.audio_path = audio_path
            except Exception as e:
                logger.error(f"Failed to save TTS audio: {e}")
        
        return True, metrics
        
    except Exception as e:
        logger.error(f"PCM streaming failed: {e}")
        return False, metrics
        
    finally:
        if stream:
            stream.close()


async def stream_tts_audio(
    text: str,
    openai_client,
    request_params: dict,
    debug: bool = False,
    save_audio: bool = False,
    audio_dir: Optional[Path] = None,
    conversation_id: Optional[str] = None
) -> Tuple[bool, StreamMetrics]:
    """Stream TTS audio with progressive playback.
    
    Args:
        text: Text to convert to speech
        openai_client: OpenAI client instance
        request_params: Parameters for TTS request
        debug: Enable debug logging
        
    Returns:
        Tuple of (success, metrics)
    """
    format = request_params.get('response_format', 'pcm')
    logger.info(f"Starting streaming TTS with format: {format}")
    
    # PCM is best for streaming (no decoding needed)
    # For other formats, we may need buffering
    if format == 'pcm':
        return await stream_pcm_audio(
            text=text,
            openai_client=openai_client,
            request_params=request_params,
            debug=debug,
            save_audio=save_audio,
            audio_dir=audio_dir,
            conversation_id=conversation_id
        )
    else:
        # Use buffered streaming for formats that need decoding
        return await stream_with_buffering(
            text=text,
            openai_client=openai_client,
            request_params=request_params,
            debug=debug,
            save_audio=save_audio,
            audio_dir=audio_dir,
            conversation_id=conversation_id
        )


# Fallback for complex formats - buffer and decode complete file
async def stream_with_buffering(
    text: str,
    openai_client,
    request_params: dict,
    sample_rate: int = 24000,  # TTS standard sample rate
    debug: bool = False,
    save_audio: bool = False,
    audio_dir: Optional[Path] = None,
    conversation_id: Optional[str] = None
) -> Tuple[bool, StreamMetrics]:
    """Fallback streaming that buffers enough data to decode reliably.

    This is used for formats like MP3, Opus, etc where frame boundaries are critical.
    Uses platform-optimized settings for stable playback.
    """
    format = request_params.get('response_format', 'pcm')
    platform_config = get_audio_config()
    logger.info(f"Using buffered streaming for format: {format} (platform: {platform_config.platform_name})")

    metrics = StreamMetrics()
    start_time = time.perf_counter()

    # Buffer for accumulating chunks
    buffer = io.BytesIO()
    # Separate buffer for saving complete audio
    save_buffer = io.BytesIO() if save_audio else None
    audio_started = False
    stream = None

    try:
        # Setup sounddevice stream with platform-optimized settings
        stream = sd.OutputStream(
            samplerate=sample_rate,
            channels=1,
            dtype='float32',
            blocksize=platform_config.blocksize
        )
        stream.start()
        
        # Use the streaming response API for true HTTP streaming
        async with openai_client.audio.speech.with_streaming_response.create(
            **request_params
        ) as response:
            first_chunk_time = None

            # Stream chunks as they arrive with platform-optimized chunk size
            async for chunk in response.iter_bytes(chunk_size=platform_config.stream_chunk_size):
                if chunk:
                    # Track first chunk for TTFA
                    if first_chunk_time is None:
                        first_chunk_time = time.perf_counter()
                        metrics.ttfa = first_chunk_time - start_time
                        logger.info(f"First chunk received - TTFA: {metrics.ttfa:.3f}s")
                    
                    buffer.write(chunk)
                    metrics.chunks_received += 1
                    
                    # Also accumulate in save buffer if saving is enabled
                    if save_buffer:
                        save_buffer.write(chunk)
                    
                    # Try to decode when we have enough data (e.g., 32KB)
                    if buffer.tell() > 32768 and not audio_started:
                        buffer.seek(0)
                        try:
                            # Attempt to decode what we have
                            audio = AudioSegment.from_file(buffer, format=format)
                            samples = np.array(audio.get_array_of_samples()).astype(np.float32) / 32768.0
                            
                            # Start playback
                            metrics.ttfa = time.perf_counter() - start_time
                            audio_started = True
                            logger.info(f"Buffered streaming started - TTFA: {metrics.ttfa:.3f}s")
                            
                            # Play audio
                            stream.write(samples)
                            metrics.chunks_played += len(samples) // 1024
                            
                            # Reset buffer for next batch
                            buffer = io.BytesIO()
                            
                        except Exception as e:
                            # Not enough valid data yet
                            buffer.seek(0, io.SEEK_END)
        
        # Process any remaining data
        if buffer.tell() > 0:
            buffer.seek(0)
            try:
                audio = AudioSegment.from_file(buffer, format=format)
                samples = np.array(audio.get_array_of_samples()).astype(np.float32) / 32768.0
                
                if not audio_started:
                    metrics.ttfa = time.perf_counter() - start_time
                    
                stream.write(samples)
                metrics.chunks_played += len(samples) // 1024
                
            except Exception as e:
                logger.error(f"Failed to decode final buffer: {e}")
        
        metrics.generation_time = time.perf_counter() - start_time
        metrics.playback_time = metrics.generation_time  # Approximate
        
        # Save audio if enabled
        if save_audio and save_buffer and audio_dir:
            try:
                from .core import save_debug_file
                save_buffer.seek(0)
                audio_data = save_buffer.read()
                audio_path = save_debug_file(audio_data, "tts", format, audio_dir, True, conversation_id)
                if audio_path:
                    logger.info(f"TTS audio saved to: {audio_path}")
            except Exception as e:
                logger.error(f"Failed to save TTS audio: {e}")
        
        return True, metrics
        
    except Exception as e:
        logger.error(f"Buffered streaming failed: {e}")
        return False, metrics
        
    finally:
        if stream:
            stream.stop()
            stream.close()