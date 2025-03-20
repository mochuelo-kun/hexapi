import numpy as np
import sounddevice as sd
import soundfile as sf
import logging
import os
import tempfile
from typing import Optional, Tuple
import curses
import queue
from contextlib import contextmanager

# Default audio settings
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_MIC_DURATION = 5.0
DEFAULT_FILE_FORMAT = "wav"

logger = logging.getLogger('chrysalis.audio_utils')

def load_audio_file(file_path: str, sample_rate: Optional[int] = None) -> Tuple[np.ndarray, int]:
    """Load audio from file into numpy array
    
    Args:
        file_path: Path to audio file
        sample_rate: Target sample rate (if None, uses file's original sample rate)
        
    Returns:
        Tuple of (audio_data, sample_rate)
    """
    logger.debug(f"Loading audio from {file_path}")
    try:
        audio_data, original_sample_rate = sf.read(file_path)
        
        # Convert to mono if stereo
        if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
            audio_data = audio_data.mean(axis=1)
        
        # Resample if needed
        if sample_rate is not None and sample_rate != original_sample_rate:
            logger.debug(f"Resampling audio from {original_sample_rate}Hz to {sample_rate}Hz")
            from scipy import signal
            num_samples = int(len(audio_data) * sample_rate / original_sample_rate)
            audio_data = signal.resample(audio_data, num_samples)
            output_sample_rate = sample_rate
        else:
            output_sample_rate = original_sample_rate
            
        logger.debug(f"Loaded audio: {len(audio_data)/output_sample_rate:.2f}s, {output_sample_rate}Hz")
        return audio_data, output_sample_rate
        
    except Exception as e:
        logger.error(f"Error loading audio file {file_path}: {e}")
        raise

def record_audio_from_microphone(duration: Optional[float] = None,
                               sample_rate: int = DEFAULT_SAMPLE_RATE,
                               interactive: bool = False) -> Tuple[np.ndarray, int]:
    """Record audio from microphone
    
    Args:
        duration: Recording duration in seconds (None for interactive mode)
        sample_rate: Sample rate in Hz
        interactive: If True, use interactive recording mode (spacebar to start/stop)
        
    Returns:
        Tuple of (audio_data, sample_rate)
    """
    if not interactive and duration is None:
        duration = DEFAULT_MIC_DURATION
        
    if interactive:
        return _record_interactive(sample_rate)
    else:
        return _record_fixed_duration(duration, sample_rate)

def _record_fixed_duration(duration: float, sample_rate: int) -> Tuple[np.ndarray, int]:
    """Record audio for a fixed duration"""
    logger.info(f"Recording {duration}s of audio at {sample_rate}Hz...")
    
    try:
        audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
        sd.wait()  # Wait until recording is finished
        audio_data = audio_data.flatten()  # Convert to 1D array
        
        logger.debug(f"Recorded {len(audio_data)/sample_rate:.2f}s of audio")
        return audio_data, sample_rate
        
    except Exception as e:
        logger.error(f"Error recording audio from microphone: {e}")
        raise

def _record_interactive(sample_rate: int) -> Tuple[np.ndarray, int]:
    """Record audio interactively using spacebar to start/stop"""
    # Create a queue to store audio chunks
    audio_queue = queue.Queue()
    recording = False
    done = False
    
    def audio_callback(indata, frames, time, status):
        """Callback for audio input"""
        if status:
            logger.warning(f"Audio input status: {status}")
        if recording:
            audio_queue.put(indata.copy())
    
    @contextmanager
    def setup_curses():
        """Setup and cleanup curses"""
        stdscr = curses.initscr()
        curses.noecho()  # Don't echo keypresses
        curses.cbreak()  # React to keys instantly
        stdscr.nodelay(1)  # Non-blocking input
        
        try:
            yield stdscr
        finally:
            # Cleanup
            curses.nocbreak()
            curses.echo()
            curses.endwin()
    
    try:
        with setup_curses() as stdscr:
            # Clear screen and show instructions
            # stdscr.clear()
            stdscr.addstr(0, 0, "Press SPACE to start recording, press SPACE again to stop...")
            stdscr.addstr(1, 0, "Press Ctrl+C to cancel")
            stdscr.refresh()
            
            # Setup audio stream
            with sd.InputStream(samplerate=sample_rate,
                              channels=1,
                              callback=audio_callback):
                while not done:
                    try:
                        # Check for spacebar press (ASCII 32)
                        c = stdscr.getch()
                        if c == ord(' '):
                            if not recording:
                                # Start recording
                                recording = True
                                stdscr.addstr(2, 0, "Recording... (Press SPACE to stop)")
                                stdscr.clrtoeol()
                            else:
                                # Stop recording
                                recording = False
                                done = True
                            stdscr.refresh()
                        
                        # Small sleep to prevent CPU hogging
                        sd.sleep(10)
                        
                    except KeyboardInterrupt:
                        done = True
                        break
        
        # Combine all audio chunks
        audio_chunks = []
        while not audio_queue.empty():
            audio_chunks.append(audio_queue.get())
        
        if not audio_chunks:
            raise ValueError("No audio was recorded")
            
        audio_data = np.concatenate(audio_chunks)
        audio_data = audio_data.flatten()
        
        logger.debug(f"Recorded {len(audio_data)/sample_rate:.2f}s of audio")
        return audio_data, sample_rate
        
    except Exception as e:
        logger.error(f"Error during interactive recording: {e}")
        raise

def save_audio_file(audio_data: np.ndarray, output_path: str, sample_rate: int = DEFAULT_SAMPLE_RATE) -> str:
    """Save audio data to file
    
    Args:
        audio_data: Audio data as numpy array
        output_path: Path to save audio file
        sample_rate: Sample rate in Hz
        
    Returns:
        Path to saved file
    """
    logger.debug(f"Saving audio to {output_path} ({len(audio_data)/sample_rate:.2f}s, {sample_rate}Hz)")
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    try:
        sf.write(output_path, audio_data, sample_rate)
        return output_path
        
    except Exception as e:
        logger.error(f"Error saving audio to {output_path}: {e}")
        raise

def resample_audio_if_needed(audio_data: np.ndarray, 
                            current_sample_rate: int, 
                            target_sample_rate: Optional[int] = None) -> Tuple[np.ndarray, int]:
    """Resample audio data if target sample rate is different from current
    
    Args:
        audio_data: Audio data as numpy array
        current_sample_rate: Current sample rate of the audio data
        target_sample_rate: Target sample rate (if None, no resampling is done)
        
    Returns:
        Tuple of (resampled_audio_data, new_sample_rate)
    """
    if target_sample_rate is None or current_sample_rate == target_sample_rate:
        return audio_data, current_sample_rate
        
    logger.debug(f"Resampling audio from {current_sample_rate}Hz to {target_sample_rate}Hz")
    from scipy import signal
    num_samples = int(len(audio_data) * target_sample_rate / current_sample_rate)
    resampled_audio = signal.resample(audio_data, num_samples)
    
    return resampled_audio, target_sample_rate

def get_audio_array(
    audio_file: Optional[str] = None,
    use_microphone: bool = False,
    microphone_interactive: bool = False,
    microphone_duration: float = DEFAULT_MIC_DURATION,
    target_sample_rate: Optional[int] = None,
    audio_array: Optional[np.ndarray] = None,
    array_sample_rate: Optional[int] = None
) -> Tuple[np.ndarray, int]:
    """Get audio data from various sources
    
    This function handles multiple audio input types:
    1. Direct numpy array input
    2. Audio file
    3. Microphone recording
    
    Args:
        audio_file: Path to audio file
        use_microphone: Whether to record from microphone
        microphone_duration: Recording duration if using microphone (seconds)
        target_sample_rate: Target sample rate for output (if None, uses original sample rate)
        audio_array: Directly provided audio data as numpy array
        array_sample_rate: Sample rate of the provided audio array
        
    Returns:
        Tuple of (audio_data, sample_rate)
        
    Raises:
        ValueError: If no valid audio input is provided or if inconsistent parameters are given
    """
    # Direct array input takes precedence
    if audio_array is not None:
        if array_sample_rate is None:
            raise ValueError("array_sample_rate must be provided when audio_array is given")
            
        logger.debug(f"Using provided audio array: {len(audio_array)/array_sample_rate:.2f}s at {array_sample_rate}Hz")
        
        # Resample if needed
        return resample_audio_if_needed(audio_array, array_sample_rate, target_sample_rate)
        
    # Record from microphone
    elif use_microphone:
        return record_audio_from_microphone(
            duration=microphone_duration, 
            interactive=microphone_interactive,
            sample_rate=target_sample_rate or DEFAULT_SAMPLE_RATE
        )
        
    # Load from file
    elif audio_file:
        return load_audio_file(audio_file, sample_rate=target_sample_rate)
        
    # No valid input provided
    else:
        raise ValueError("Either audio_file, use_microphone, or audio_array must be provided")

def audio_array_to_file(
    audio_data: np.ndarray,
    sample_rate: int,
    output_path: Optional[str] = None,
    file_format: str = DEFAULT_FILE_FORMAT
) -> str:
    """Convert audio array to file, optionally saving to a specified path
    
    Args:
        audio_data: Audio data as numpy array
        sample_rate: Sample rate of the audio data in Hz
        output_path: Path to save audio file (if None, creates a temporary file)
        file_format: Audio format (default: wav)
        
    Returns:
        Path to the saved file
    """
    if output_path is None:
        # Create a temporary file with the specified format
        with tempfile.NamedTemporaryFile(suffix=f".{file_format}", delete=False) as temp_file:
            output_path = temp_file.name
    
    return save_audio_file(audio_data, output_path, sample_rate=sample_rate)

def play_audio(audio_data: np.ndarray, sample_rate: int = DEFAULT_SAMPLE_RATE) -> None:
    """Play audio data through system speakers
    
    Args:
        audio_data: Audio data as numpy array
        sample_rate: Sample rate of the audio data in Hz
    """
    logger.debug(f"Playing audio: {len(audio_data)/sample_rate:.2f}s at {sample_rate}Hz")
    try:
        sd.play(audio_data, sample_rate)
        sd.wait()  # Wait until playback is finished
        logger.debug("Audio playback completed")
    except Exception as e:
        logger.error(f"Error playing audio: {e}")
        raise 