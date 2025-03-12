import logging
import time
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from .base import SpeechToTextBase
from ..config import Config

logger = logging.getLogger('chrysalis.stt.sherpa_local')

# Default model selections
DEFAULT_STT_MODEL = "sherpa-onnx-zipformer-en-2023-06-26"
DEFAULT_RECOGNITION_MODEL = "wespeaker_en_voxceleb_CAM++"
DEFAULT_SEGMENTATION_MODEL = "sherpa-onnx-pyannote-segmentation-3-0"

# Model configuration
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_NUM_MEL_BINS = 80
DEFAULT_DECODING_METHOD = "greedy_search"

# Diarization configuration
DEFAULT_MIN_DURATION_ON = 0.3
DEFAULT_MIN_DURATION_OFF = 0.5
DEFAULT_CLUSTER_THRESHOLD = 0.5

class ModelFiles:
    """Helper class to find and manage model files"""
    def __init__(self, model_dir: Path, use_int8: bool = False):
        self.model_dir = model_dir
        self.use_int8 = use_int8
        self.files = list(model_dir.glob("**/*.onnx"))
        
    def _find_file(self, patterns: List[str], required: bool = True) -> Optional[Path]:
        """Find a file matching any of the patterns"""
        for pattern in patterns:
            # Try int8 version first if enabled
            if self.use_int8:
                int8_matches = [f for f in self.files if pattern in f.stem and f.stem.endswith(".int8")]
                if int8_matches:
                    return int8_matches[0]
            
            # Try regular version
            matches = [f for f in self.files if pattern in f.stem and not f.stem.endswith(".int8")]
            if matches:
                return matches[0]
                
        if required:
            patterns_str = '", "'.join(patterns)
            raise FileNotFoundError(
                f"Could not find required model file matching patterns: \"{patterns_str}\" "
                f"in {self.model_dir}"
            )
        return None
    
    def find_encoder(self) -> Optional[Path]:
        """Find encoder model file"""
        return self._find_file(["encoder", "encode"])
        
    def find_decoder(self) -> Optional[Path]:
        """Find decoder model file"""
        return self._find_file(["decoder", "decode"])
        
    def find_joiner(self) -> Optional[Path]:
        """Find joiner model file"""
        return self._find_file(["joiner"], required=False)
        
    def find_single_model(self) -> Optional[Path]:
        """Find single model file"""
        return self._find_file(["model"])
        
    def find_tokens(self) -> Path:
        """Find tokens file"""
        # Check for tokens.txt in model dir
        tokens_file = self.model_dir / "tokens.txt"
        if tokens_file.exists():
            return tokens_file
            
        # Check for model-specific tokens file
        model_tokens = list(self.model_dir.glob("*tokens.txt"))
        if model_tokens:
            return model_tokens[0]
            
        raise FileNotFoundError(f"Could not find tokens file in {self.model_dir}")

class SherpaLocalSTT(SpeechToTextBase):
    def __init__(self, 
                 model_name: str = DEFAULT_STT_MODEL,
                 enable_diarization: bool = False,
                 recognition_model: Optional[str] = None,
                 segmentation_model: Optional[str] = None,
                 use_int8: bool = False):
        """
        Initialize Sherpa-ONNX local STT with optional diarization
        
        Args:
            model_name: Name of the STT model to use
            enable_diarization: Whether to enable speaker diarization
            recognition_model: Optional custom recognition model name
            segmentation_model: Optional custom segmentation model name
            use_int8: Whether to use int8 quantized models if available
        """
        self.model_name = model_name
        self.enable_diarization = enable_diarization
        self.recognition_model = recognition_model or DEFAULT_RECOGNITION_MODEL
        self.segmentation_model = segmentation_model or DEFAULT_SEGMENTATION_MODEL
        self.use_int8 = use_int8
        
        try:
            import sherpa_onnx
            
            # Initialize recognizer
            self.recognizer = self._init_recognizer()
            
            # Initialize diarizer if enabled
            self.diarizer = self._init_diarizer() if enable_diarization else None
            
            logger.info("Sherpa-ONNX STT initialized successfully with model: %s", model_name)
            if enable_diarization:
                logger.info("Speaker diarization enabled with models:")
                logger.info("  Recognition: %s", self.recognition_model)
                logger.info("  Segmentation: %s", self.segmentation_model)
                
        except ImportError as e:
            logger.error("Failed to import Sherpa-ONNX dependencies: %s", e)
            raise ImportError(
                "Missing required dependency: sherpa-onnx. "
                "Please check the README for installation instructions."
            )
        
    def _init_recognizer(self):
        """Initialize the speech recognizer"""
        import sherpa_onnx
        
        model_dir = Path(Config.ONNX_MODEL_DIR) / "stt" / self.model_name
        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
            
        model_files = ModelFiles(model_dir, self.use_int8)
        
        # Try to detect model type and use appropriate factory method
        
        # Check for Whisper model pattern (encoder + decoder)
        if any("whisper" in str(f).lower() for f in model_files.files):
            encoder = model_files.find_encoder()
            decoder = model_files.find_decoder()
            tokens = model_files.find_tokens()
            
            logger.info("Using Whisper model configuration:")
            logger.info("  Encoder: %s", encoder.name)
            logger.info("  Decoder: %s", decoder.name)
            
            return sherpa_onnx.OfflineRecognizer.from_whisper(
                encoder=str(encoder),
                decoder=str(decoder),
                tokens=str(tokens),
                num_threads=1,
                debug=True
            )
        
        # Check for Transducer model pattern (encoder + decoder + joiner)
        encoder = model_files.find_encoder()
        if encoder and "moonshine" not in str(encoder).lower():
            decoder = model_files.find_decoder()
            joiner = model_files.find_joiner()
            tokens = model_files.find_tokens()
            
            logger.info("Using Transducer model configuration:")
            logger.info("  Encoder: %s", encoder.name)
            logger.info("  Decoder: %s", decoder.name)
            if joiner:
                logger.info("  Joiner: %s", joiner.name)
            
            return sherpa_onnx.OfflineRecognizer.from_transducer(
                encoder=str(encoder),
                decoder=str(decoder),
                joiner=str(joiner) if joiner else None,
                tokens=str(tokens),
                num_threads=1,
                debug=True
            )
        
        # Check for Moonshine model pattern
        if any("moonshine" in str(f).lower() for f in model_files.files):
            preprocessor = model_files._find_file(["preprocess"])
            encoder = model_files.find_encoder()
            uncached_decoder = model_files._find_file(["uncached_decode"])
            cached_decoder = model_files._find_file(["cached_decode"])
            tokens = model_files.find_tokens()
            
            logger.info("Using Moonshine model configuration:")
            logger.info("  Preprocessor: %s", preprocessor.name)
            logger.info("  Encoder: %s", encoder.name)
            logger.info("  Uncached Decoder: %s", uncached_decoder.name)
            logger.info("  Cached Decoder: %s", cached_decoder.name)
            
            return sherpa_onnx.OfflineRecognizer.from_moonshine(
                preprocessor=str(preprocessor),
                encoder=str(encoder),
                uncached_decoder=str(uncached_decoder),
                cached_decoder=str(cached_decoder),
                tokens=str(tokens),
                num_threads=1,
                debug=True
            )
        
        # Check for single CTC model file (paraformer, nemo-ctc, etc.)
        model = model_files.find_single_model()
        if model:
            tokens = model_files.find_tokens()
            
            logger.info("Using CTC model configuration:")
            logger.info("  Model: %s", model.name)
            
            # Try to detect specific model type from path
            if "paraformer" in str(model).lower():
                return sherpa_onnx.OfflineRecognizer.from_paraformer(
                    model=str(model),
                    tokens=str(tokens),
                    num_threads=1,
                    debug=True
                )
            elif "nemo" in str(model).lower() and "ctc" in str(model).lower():
                return sherpa_onnx.OfflineRecognizer.from_nemo_ctc(
                    model=str(model),
                    tokens=str(tokens),
                    num_threads=1,
                    debug=True
                )
            else:
                # Default to wenet-ctc for other single-file models
                return sherpa_onnx.OfflineRecognizer.from_wenet_ctc(
                    model=str(model),
                    tokens=str(tokens),
                    num_threads=1,
                    debug=True
                )
        
        raise ValueError(
            f"Could not determine model type from files in {model_dir}. "
            "Please check the model directory contains the correct files."
        )
        
    def _init_diarizer(self):
        """Initialize the speaker diarizer with custom models"""
        import sherpa_onnx
        
        model_dir = Path(Config.ONNX_MODEL_DIR) / "diarization"
        
        # Find recognition model
        recog_dir = model_dir / "recognition"
        recog_files = ModelFiles(recog_dir, self.use_int8)
        embedding_model = recog_files._find_file([self.recognition_model])
        
        # Find segmentation model
        seg_dir = model_dir / "segmentation" / self.segmentation_model
        seg_files = ModelFiles(seg_dir, self.use_int8)
        segmentation_model = seg_files.find_single_model()
        
        logger.info("Using diarization models:")
        logger.info("  Recognition: %s", embedding_model.name)
        logger.info("  Segmentation: %s", segmentation_model.name)
            
        config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
            segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
                pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                    model=str(segmentation_model)
                ),
            ),
            embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                model=str(embedding_model)
            ),
            clustering=sherpa_onnx.FastClusteringConfig(
                num_clusters=-1,  # Auto-detect number of speakers
                threshold=DEFAULT_CLUSTER_THRESHOLD
            ),
            min_duration_on=DEFAULT_MIN_DURATION_ON,
            min_duration_off=DEFAULT_MIN_DURATION_OFF,
        )
        
        return sherpa_onnx.OfflineSpeakerDiarization(config)
    
    def transcribe_file(self, audio_file: str):
        """Transcribe audio from a file with optional diarization"""
        logger.debug("Transcribing file with Sherpa-ONNX: %s", audio_file)
        start = time.time()
        
        # Load audio
        audio_waveform, sample_rate = sf.read(audio_file, dtype="float32", always_2d=True)
        audio_waveform = audio_waveform[:, 0]  # only use the first channel
        
        # Get transcription
        stream = self.recognizer.create_stream()
        stream.accept_waveform(sample_rate, audio_waveform)
        self.recognizer.decode_stream(stream)
        text = stream.result.text.strip()
        
        # Handle diarization if enabled
        if self.enable_diarization and self.diarizer:
            # Process for diarization
            logger.debug("Running speaker diarization")
            speaker_segments = self.diarizer.process(audio_waveform).sort_by_start_time()
            logger.debug("Found %d speaker segments", len(speaker_segments))
            
            # Debug: Check what's available in the objects
            result = stream.result
            
            # Now we can align tokens and timestamps with speaker segments
            if hasattr(result, 'tokens') and hasattr(result, 'timestamps'):
                tokens = result.tokens
                timestamps = result.timestamps
                
                # Create a mapping of token index to speaker based on timestamp
                token_to_speaker = {}
                
                # For each token, find which speaker segment it belongs to
                for i, timestamp in enumerate(timestamps):
                    if i >= len(tokens):
                        break
                        
                    # Find the speaker segment that contains this timestamp
                    assigned_speaker = None
                    for segment in speaker_segments:
                        if segment.start <= timestamp < segment.end:
                            assigned_speaker = segment.speaker
                            break
                            
                    # If no speaker found, use the closest one
                    if assigned_speaker is None:
                        min_distance = float('inf')
                        for segment in speaker_segments:
                            # Distance to segment start or end, whichever is closer
                            distance = min(abs(timestamp - segment.start), abs(timestamp - segment.end))
                            if distance < min_distance:
                                min_distance = distance
                                assigned_speaker = segment.speaker
                    
                    token_to_speaker[i] = assigned_speaker
                
                # Group consecutive tokens by speaker
                segments = []
                current_speaker = None
                current_segment = None
                
                for i, token in enumerate(tokens):
                    if i >= len(timestamps):
                        break
                        
                    speaker = token_to_speaker.get(i)
                    if speaker != current_speaker:
                        # Start a new segment
                        if current_segment:
                            segments.append(current_segment)
                        
                        current_speaker = speaker
                        current_segment = {
                            "speaker": speaker,
                            "start": timestamps[i],
                            "tokens": [token],
                            "token_indices": [i]
                        }
                    else:
                        # Add to current segment
                        current_segment["tokens"].append(token)
                        current_segment["token_indices"].append(i)
                
                # Add the last segment
                if current_segment:
                    segments.append(current_segment)
                
                # For each segment, get the text and end time
                diarized_text = {
                    "text": text,
                    "segments": []
                }
                
                for segment in segments:
                    # Get all tokens in this segment
                    segment_text = "".join(segment["tokens"]).strip()
                    
                    # Get the end time from the last token in the segment
                    last_token_idx = segment["token_indices"][-1]
                    if last_token_idx + 1 < len(timestamps):
                        segment["end"] = timestamps[last_token_idx + 1]
                    else:
                        # Use the speaker segment end time as fallback
                        for spk_segment in speaker_segments:
                            if spk_segment.speaker == segment["speaker"] and spk_segment.start <= segment["start"] < spk_segment.end:
                                segment["end"] = spk_segment.end
                                break
                        else:
                            # Last resort fallback
                            segment["end"] = segment["start"] + 1.0
                    
                    # Add to result
                    if segment_text:  # Only add non-empty segments
                        diarized_text["segments"].append({
                            "speaker": segment["speaker"],
                            "start": segment["start"],
                            "end": segment["end"],
                            "text": segment_text
                        })
                
                return diarized_text
            else:
                # Fallback to the previous approach
                logger.debug("Tokens or timestamps not available, using sentence-based approach")
                diarized_result = self._align_transcription_with_speakers_fallback(text, speaker_segments)
                return diarized_result
        
        duration = time.time() - start
        logger.debug("File transcription completed in %.2fs", duration)
        return text
    
    def transcribe_audio(self, audio_data: np.ndarray, sample_rate: int) -> str:
        """Transcribe audio from numpy array with optional diarization"""
        logger.debug("Transcribing audio data with Sherpa-ONNX")
        start = time.time()
        
        # Create stream and process audio
        stream = self.recognizer.create_stream()
        stream.accept_waveform_from_numpy(audio_data, sample_rate)
        
        # Get transcription
        self.recognizer.decode_stream(stream)
        text = stream.result.text.strip()
        
        # Handle diarization if enabled
        if self.enable_diarization and self.diarizer:
            segments = self.diarizer.diarize_numpy(audio_data, sample_rate)
            text = self._format_diarized_text(text, segments)
        
        duration = time.time() - start
        logger.debug("Audio data transcription completed in %.2fs", duration)
        return text
    

    def _align_transcription_with_speakers(self, text, speaker_segments):
        """Align transcription with speaker segments based on timestamps"""
        # Initialize result structure
        diarized_text = {
            "text": text,
            "segments": []
        }
        
        # Get the recognizer result object (need to be called right after decode_stream)
        # This approach won't work for transcribe_audio since we don't have the result object
        # Let's work with the tokens and timestamps arrays directly
        
        # Check if we have the necessary attributes in the recognizer result
        if hasattr(stream.result, 'tokens') and hasattr(stream.result, 'timestamps'):
            tokens = stream.result.tokens
            timestamps = stream.result.timestamps
            
            # Check if we have enough tokens and timestamps
            if len(tokens) > 0 and len(timestamps) > 0:
                logger.debug("Using token-based alignment with %d tokens and %d timestamps", 
                            len(tokens), len(timestamps))
                
                # Create a mapping of token index to speaker based on timestamp
                token_to_speaker = {}
                
                # For each token, find which speaker segment it belongs to
                for i, timestamp in enumerate(timestamps):
                    if i >= len(tokens):
                        break
                        
                    # Find the speaker segment that contains this timestamp
                    assigned_speaker = None
                    for segment in speaker_segments:
                        if segment.start <= timestamp < segment.end:
                            assigned_speaker = segment.speaker
                            break
                            
                    # If no speaker found, use the closest one
                    if assigned_speaker is None:
                        min_distance = float('inf')
                        for segment in speaker_segments:
                            # Distance to segment start or end, whichever is closer
                            distance = min(abs(timestamp - segment.start), abs(timestamp - segment.end))
                            if distance < min_distance:
                                min_distance = distance
                                assigned_speaker = segment.speaker
                    
                    token_to_speaker[i] = assigned_speaker
                
                # Group consecutive tokens by speaker
                segments = []
                current_speaker = None
                current_segment = None
                
                for i, token in enumerate(tokens):
                    if i >= len(timestamps):
                        break
                        
                    speaker = token_to_speaker.get(i)
                    if speaker != current_speaker:
                        # Start a new segment
                        if current_segment:
                            segments.append(current_segment)
                        
                        current_speaker = speaker
                        current_segment = {
                            "speaker": speaker,
                            "start": timestamps[i],
                            "tokens": [token],
                            "token_indices": [i]
                        }
                    else:
                        # Add to current segment
                        current_segment["tokens"].append(token)
                        current_segment["token_indices"].append(i)
                
                # Add the last segment
                if current_segment:
                    segments.append(current_segment)
                
                # For each segment, get the text and end time
                for segment in segments:
                    # Get all tokens in this segment
                    segment_text = "".join(segment["tokens"]).strip()
                    
                    # Get the end time from the last token in the segment
                    last_token_idx = segment["token_indices"][-1]
                    if last_token_idx + 1 < len(timestamps):
                        segment["end"] = timestamps[last_token_idx + 1]
                    else:
                        # Use the speaker segment end time as fallback
                        for spk_segment in speaker_segments:
                            if spk_segment.speaker == segment["speaker"] and spk_segment.start <= segment["start"] < spk_segment.end:
                                segment["end"] = spk_segment.end
                                break
                        else:
                            # Last resort fallback
                            segment["end"] = segment["start"] + 1.0
                    
                    # Add to result
                    if segment_text:  # Only add non-empty segments
                        diarized_text["segments"].append({
                            "speaker": segment["speaker"],
                            "start": segment["start"],
                            "end": segment["end"],
                            "text": segment_text
                        })
                
                # If we managed to create segments, return them
                if diarized_text["segments"]:
                    return diarized_text
        
        # Fallback to simpler approach if the above didn't work
        logger.debug("Falling back to simpler text-splitting approach")
        
        # Split the text into sentences and assign to speakers
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Estimate total duration from the last segment's end time
        total_duration = speaker_segments[-1].end if speaker_segments else 0
        if not total_duration:
            logger.warning("Could not determine audio duration from segments")
            return {"text": text, "segments": [{"speaker": "Unknown", "start": 0, "end": 0, "text": text}]}
        
        # Calculate rough timestamp for each sentence based on character count
        char_per_second = len(text) / total_duration if total_duration > 0 else 1
        
        # Create sentence segments with estimated timestamps
        sentence_segments = []
        current_pos = 0
        current_time = 0
        
        for sentence in sentences:
            if not sentence.strip():
                continue
                
            sentence_len = len(sentence)
            sentence_duration = sentence_len / char_per_second
            
            sentence_segments.append({
                "text": sentence,
                "start": current_time,
                "end": current_time + sentence_duration
            })
            
            current_pos += sentence_len
            current_time += sentence_duration
        
        # Now assign speakers to each sentence based on time overlap
        for sentence in sentence_segments:
            # Find which speaker segment has the most overlap with this sentence
            max_overlap = 0
            best_speaker = 0
            
            for segment in speaker_segments:
                # Calculate overlap between sentence and speaker segment
                overlap_start = max(sentence["start"], segment.start)
                overlap_end = min(sentence["end"], segment.end)
                overlap = max(0, overlap_end - overlap_start)
                
                if overlap > max_overlap:
                    max_overlap = overlap
                    best_speaker = segment.speaker
            
            # Add this sentence with its assigned speaker
            diarized_text["segments"].append({
                "speaker": best_speaker,
                "start": sentence["start"],
                "end": sentence["end"],
                "text": sentence["text"]
            })
        
        return diarized_text

    def _align_transcription_with_speakers_fallback(self, text, speaker_segments):
        """Fallback alignment method when token timestamps aren't available"""
        # Initialize result structure
        diarized_text = {
            "text": text,
            "segments": []
        }
        
        # Split the text into sentences
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Estimate total duration from the last segment's end time
        total_duration = speaker_segments[-1].end if speaker_segments else 0
        if not total_duration:
            logger.warning("Could not determine audio duration from segments")
            return {"text": text, "segments": [{"speaker": "Unknown", "start": 0, "end": 0, "text": text}]}
        
        # Create sentence segments with estimated timestamps
        char_per_second = len(text) / total_duration if total_duration > 0 else 1
        sentence_segments = []
        current_time = 0
        
        for sentence in sentences:
            if not sentence.strip():
                continue
                
            sentence_len = len(sentence)
            sentence_duration = sentence_len / char_per_second
            
            sentence_segments.append({
                "text": sentence,
                "start": current_time,
                "end": current_time + sentence_duration
            })
            
            current_time += sentence_duration
        
        # Assign speakers based on overlap
        for sentence in sentence_segments:
            # Find which speaker segment has the most overlap
            max_overlap = 0
            best_speaker = 0
            
            for segment in speaker_segments:
                overlap_start = max(sentence["start"], segment.start)
                overlap_end = min(sentence["end"], segment.end)
                overlap = max(0, overlap_end - overlap_start)
                
                if overlap > max_overlap:
                    max_overlap = overlap
                    best_speaker = segment.speaker
            
            # Add this sentence with its assigned speaker
            diarized_text["segments"].append({
                "speaker": best_speaker,
                "start": sentence["start"],
                "end": sentence["end"],
                "text": sentence["text"]
            })
        
        return diarized_text

    def _format_diarized_text(self, text, segments):
        """Format text with speaker diarization information"""
        if isinstance(segments, dict) and "segments" in segments:
            # If segments is already a properly formatted result, return it
            return segments
            
        if hasattr(segments, "segments"):
            # If segments is a result object with segments attribute
            segments = segments.segments
        
        diarized_result = {
            "text": text,
            "segments": []
        }
        
        for segment in segments:
            if hasattr(segment, "speaker") and hasattr(segment, "start") and hasattr(segment, "end"):
                # Handle proper segment objects
                segment_text = ""
                if hasattr(segment, "text") and segment.text:
                    segment_text = segment.text
                
                diarized_result["segments"].append({
                    "speaker": segment.speaker,
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment_text if segment_text else f"[Segment from {segment.start:.2f}s to {segment.end:.2f}s]"
                })
        
        # If we have no segments, add a default one
        if not diarized_result["segments"]:
            logger.warning("No speaker segments found in diarization results. Using full text.")
            diarized_result["segments"].append({
                "speaker": "Unknown",
                "start": 0,
                "end": 0,
                "text": text
            })
        
        return diarized_result