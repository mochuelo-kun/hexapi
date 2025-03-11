import click
import json
from pathlib import Path
from .pipeline import Pipeline
from .config import Config

@click.group()
def cli():
    """Chrysalis CLI - Audio Processing Pipeline"""
    pass

@cli.command()
@click.argument('audio_input', type=click.Path(exists=True))
@click.option('--enable-diarization', is_flag=True, help='Enable speaker diarization')
@click.option('--recognition-model', help='Speaker recognition model name (requires --enable-diarization)')
@click.option('--segmentation-model', help='Speaker segmentation model name (requires --enable-diarization)')
@click.option('--use-int8', is_flag=True, help='Use int8 quantized models if available')
@click.option('--output', '-o', type=click.Path(), help='Output file for transcription')
@click.option('--format', '-f', type=click.Choice(['text', 'json']), default='text',
              help='Output format (text or json)')
def transcribe(audio_input: str, enable_diarization: bool, recognition_model: str,
               segmentation_model: str, use_int8: bool, output: str, format: str):
    """Transcribe audio file to text with optional diarization
    
    Models should be placed in {ONNX_MODEL_DIR}/diarization/{recognition,segmentation}/
    """
    # Validate diarization options
    if (recognition_model or segmentation_model) and not enable_diarization:
        raise click.UsageError("--recognition-model and --segmentation-model require --enable-diarization")
    
    # Initialize pipeline with diarization options
    pipeline = Pipeline(
        enable_diarization=enable_diarization,
        stt_params={
            "recognition_model": recognition_model,
            "segmentation_model": segmentation_model,
            "use_int8": use_int8
        } if enable_diarization else {"use_int8": use_int8}
    )
    
    result = pipeline.transcribe_to_text(audio_input)
    
    # Format output
    if format == 'json':
        if not enable_diarization:
            result = {"text": result}
        output_text = json.dumps(result, indent=2)
    else:
        if enable_diarization:
            # Format diarized text nicely
            segments = result["segments"]
            lines = []
            for seg in segments:
                lines.append(f"[Speaker {seg['speaker']}] {seg['text']}")
            output_text = "\n".join(lines)
        else:
            output_text = result
            
    # Write or print output
    if output:
        Path(output).write_text(output_text)
        click.echo(f"Transcription saved to: {output}")
    else:
        click.echo(output_text)

@cli.command()
@click.argument('audio_input', type=click.Path(exists=True))
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
def query(audio_input: str, enable_diarization: bool, recognition_model: str,
          segmentation_model: str, use_int8: bool, system_prompt: str, 
          output_audio: str, speaker_id: int, speed: float, format: str):
    """Process audio query through the full pipeline with optional diarization
    
    Models should be placed in {ONNX_MODEL_DIR}/diarization/{recognition,segmentation}/
    """
    # Validate diarization options
    if (recognition_model or segmentation_model) and not enable_diarization:
        raise click.UsageError("--recognition-model and --segmentation-model require --enable-diarization")
    
    # Initialize pipeline with diarization options
    pipeline = Pipeline(
        enable_diarization=enable_diarization,
        stt_params={
            "recognition_model": recognition_model,
            "segmentation_model": segmentation_model,
            "use_int8": use_int8
        } if enable_diarization else {"use_int8": use_int8}
    )
    
    result = pipeline.process_audio_query(
        audio_input,
        system_prompt=system_prompt,
        output_audio=output_audio,
        speaker_id=speaker_id,
        speed=speed
    )
    
    # Format output
    if format == 'json':
        click.echo(json.dumps(result, indent=2))
    else:
        click.echo("\nTranscription:")
        if enable_diarization:
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