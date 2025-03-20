import click
import json
import logging
from pathlib import Path
from .pipeline import Pipeline, PipelineConfig
from .config import Config
from .logging import setup_logging
from typing import Optional

# Initialize logging at the module level
logger = setup_logging(level=logging.DEBUG)

@click.group()
def cli():
    """Chrysalis CLI - Audio Processing Pipeline"""
    pass

@cli.command()
@click.argument('audio-input-file', type=click.Path(exists=True), required=False)
@click.option('--use-microphone', '-m', is_flag=True, help='Record audio from microphone')
@click.option('--microphone-interactive', '-m', is_flag=True, help='Whether microphone is triggered on user input')
@click.option('--microphone-duration', '-d', type=float, help='Recording microphone_duration in seconds')
@click.option('--enable-diarization', is_flag=True, help='Enable speaker diarization')
@click.option('--recognition-model', help='Speaker recognition model name (requires --enable-diarization)')
@click.option('--segmentation-model', help='Speaker segmentation model name (requires --enable-diarization)')
@click.option('--num-speakers', type=int, help='Number of speakers present (requires --enable-diarization)')
@click.option('--use-int8', is_flag=True, help='Use int8 quantized models if available')
@click.option('--output-file', '-o', type=click.Path(), help='Output file for transcription')
@click.option('--format', '-f', type=click.Choice(['text', 'json']), default='text',
              help='Output format (text or json)')
def transcribe(
    audio_input_file: Optional[str],
    use_microphone: bool,
    microphone_interactive: bool,
    microphone_duration: Optional[float],
    enable_diarization: bool,
    recognition_model: str,
    segmentation_model: str,
    num_speakers: int,
    use_int8: bool,
    output_file: str,
    format: str,
):
    """Transcribe audio file or microphone input to text with optional diarization"""
    # Input validation
    if not audio_input_file and not use_microphone:
        raise click.UsageError("Either provide an audio file or use --use-microphone")
    if audio_input_file and use_microphone:
        raise click.UsageError("Cannot use both audio file and microphone input")
    if (recognition_model or segmentation_model) and not enable_diarization:
        raise click.UsageError("--recognition-model and --segmentation-model require --enable-diarization")
    
    # Initialize pipeline config
    config = PipelineConfig(
        enable_diarization=enable_diarization,
        recognition_model=recognition_model,
        segmentation_model=segmentation_model,
        num_speakers=num_speakers,
        use_int8=use_int8
    )
    
    # Initialize pipeline with configuration
    pipeline = Pipeline(config=config)
    
    try:
        # Process the audio
        transcription = pipeline.stt.transcribe(
            audio_file=audio_input_file,
            use_microphone=use_microphone,
            microphone_interactive=microphone_interactive,
            microphone_duration=microphone_duration,
        )
        
        # Format output
        if format == 'json':
            if enable_diarization and isinstance(transcription, dict):
                output_text = json.dumps(transcription, indent=2)
            else:
                output_text = json.dumps({"text": transcription}, indent=2)
        else:
            if enable_diarization and isinstance(transcription, dict) and "segments" in transcription:
                # Format diarized text nicely
                lines = []
                for segment in transcription["segments"]:
                    if isinstance(segment, dict):
                        start = segment.get("start", 0)
                        speaker = segment.get("speaker", "Unknown")
                        text = segment.get("text", "")
                        lines.append(f"[{start:.2f}s] (Speaker {speaker}) {text}")
                    else:
                        lines.append(str(segment))
                output_text = "\n".join(lines)
            else:
                output_text = transcription
                
        # Write or print output
        if output_file:
            Path(output_file).write_text(output_text)
            click.echo(f"Transcription saved to: {output_file}")
        else:
            click.echo(output_text)
            
    except KeyboardInterrupt:
        click.echo("\nRecording cancelled.")
        return
    except Exception as e:
        logger.error(f"Error during transcription: {e}")
        raise click.ClickException(str(e))

@cli.command()
@click.argument('text')
@click.option('--model', '-m', help='LLM model to use (e.g. "ollama:llama2")')
@click.option('--system-prompt', '-s', help='System prompt for LLM')
@click.option('--format', '-f', type=click.Choice(['text', 'json']), default='text',
              help='Output format (text or json)')
def query(text: str, model: str, system_prompt: str, format: str):
    """Query LLM with text input"""
    config = PipelineConfig()
    if model:
        # Determine implementation from model name
        if "openai" in model.lower():
            config.llm_implementation = "openai"
        else:
            config.llm_implementation = "ollama"
        config.llm_model = model
    
    pipeline = Pipeline(config=config)
    
    response = pipeline.query_only(text, system_prompt)
    
    if format == 'json':
        click.echo(json.dumps({
            "input": text,
            "response": response
        }, indent=2))
    else:
        click.echo(response)

@cli.command()
@click.argument('text')
@click.option('--output', '-o', type=click.Path(), help='Output audio file (if not specified, plays through speakers)')
@click.option('--tts-model', '-m', help='TTS model to use')
@click.option('--speaker-id', type=int, default=0, help='Speaker ID for multi-speaker models')
@click.option('--speed', type=float, default=1.0, help='Speech speed factor')
@click.option('--no-play', is_flag=True, help='Disable audio playback when no output file specified')
def speak(text: str, output: str, tts_model: str, speaker_id: int, speed: float, no_play: bool):
    """Synthesize text to speech and optionally save to file or play through speakers"""
    config = PipelineConfig(
        tts_model=tts_model,
        speaker_id=speaker_id,
        speed=speed
    )
    
    pipeline = Pipeline(config=config)
    
    pipeline.synthesize_only(
        text, 
        output_file=output,
        play_audio=not no_play
    )
    
    if output:
        click.echo(f"Audio saved to: {output}")
    elif not no_play:
        click.echo("Playing audio through speakers...")

@cli.command()
@click.argument('audio-input-file', type=click.Path(exists=True))
@click.option('--enable-diarization', is_flag=True, help='Enable speaker diarization')
@click.option('--recognition-model', help='Speaker recognition model name (requires --enable-diarization)')
@click.option('--segmentation-model', help='Speaker segmentation model name (requires --enable-diarization)')
@click.option('--use-int8', is_flag=True, help='Use int8 quantized models if available')
@click.option('--system-prompt', '-s', help='System prompt for LLM')
@click.option('--output-audio', '-o', type=click.Path(), help='Generate audio response')
@click.option('--speaker-id', type=int, default=0, help='TTS speaker ID')
@click.option('--speed', type=float, default=1.0, help='TTS speech speed')
@click.option('--format', '-f', type=click.Choice(['text', 'json']), default='text',
              help='Output format (text or json)')
def pipeline(audio_input_file: str, enable_diarization: bool, recognition_model: str,
            segmentation_model: str, use_int8: bool, system_prompt: str, 
            output_audio: str, speaker_id: int, speed: float, format: str):
    """Process audio through the full pipeline (STT -> LLM -> TTS)
    
    Models should be placed in {ONNX_MODEL_DIR}/diarization/{recognition,segmentation}/
    """
    # Validate diarization options
    if (recognition_model or segmentation_model) and not enable_diarization:
        raise click.UsageError("--recognition-model and --segmentation-model require --enable-diarization")
    
    # Initialize pipeline config
    config = PipelineConfig(
        # stt_implementation=DEFAULT_STT_IMPLEMENTATION,
        enable_diarization=enable_diarization,
        recognition_model=recognition_model,
        segmentation_model=segmentation_model,
        use_int8=use_int8,
        speaker_id=speaker_id,
        speed=speed
    )
    
    # Initialize pipeline with configuration
    pipeline = Pipeline(config=config)
    
    result = pipeline.process_audio_query(
        audio_input_file,
        system_prompt=system_prompt,
        output_audio=output_audio
    )
    
    # Format output
    if format == 'json':
        click.echo(json.dumps(result, indent=2))
    else:
        click.echo("\nTranscription:")
        if enable_diarization and isinstance(result["transcription"], dict):
            for seg in result["transcription"]["segments"]:
                click.echo(f"[Speaker {seg['speaker']}] {seg['text']}")
        else:
            click.echo(result["transcription"])
            
        click.echo("\nResponse:")
        click.echo(result["response"])
        
        if result["audio_path"]:
            click.echo(f"\nAudio response saved to: {result['audio_path']}")

@cli.command()
@click.option('--model-type', '-t', 
              type=click.Choice(['stt', 'recognition', 'segmentation']),
              required=True, help='Type of model to list')
@click.option('--show-int8', is_flag=True, help='Show int8 model variants')
def list_models(model_type: str, show_int8: bool):
    """List available models of the specified type"""
    base_dir = Path(Config.ONNX_MODEL_DIR)
    
    if model_type == 'stt':
        model_dir = base_dir / "stt"
        models = [d.name for d in model_dir.iterdir() if d.is_dir()]
    else:
        model_dir = base_dir / "diarization" / model_type
        models = []
        for model in model_dir.glob("**/*.onnx"):
            if show_int8 or not model.stem.endswith(".int8"):
                models.append(model.stem.replace(".int8", ""))
    
    if not models:
        click.echo(f"No {model_type} models found in {model_dir}")
        return
        
    click.echo(f"\nAvailable {model_type} models:")
    for model in sorted(set(models)):  # Remove duplicates from int8 variants
        click.echo(f"- {model}")

if __name__ == '__main__':
    cli() 