import click
from .pipeline import ChrysalisPipeline, ChrysalisConfig
from .logging import setup_logging

logger = setup_logging()

@click.group()
def cli():
    """Voice Assistant CLI"""
    pass

@cli.command()
@click.option('--audio-file', type=str, help='Input audio file path')
@click.option('--use-mic', is_flag=True, help='Use microphone input')
@click.option('--implementation', type=str, default='whisper_local', 
              help='STT implementation to use')
def transcribe(audio_file, use_mic, implementation):
    """Transcribe speech to text"""
    config = ChrysalisConfig(stt_implementation=implementation)
    pipeline = ChrysalisPipeline(config)
    text = pipeline.transcribe_only(audio_file=audio_file, use_mic=use_mic)
    click.echo(f"Transcription: {text}")
    return text

@cli.command()
@click.argument('prompt')
@click.option('--model', default='ollama:llama2', help='LLM model to use')
def query(prompt, model):
    """Query LLM with text prompt"""
    config = ChrysalisConfig(llm_model=model)
    pipeline = ChrysalisPipeline(config)
    response = pipeline.query_only(prompt)
    click.echo(f"Response: {response}")
    return response

@cli.command()
@click.argument('text')
@click.option('--output', type=str, default='output.mp3', help='Output audio file')
@click.option('--implementation', type=str, default='elevenlabs',
              help='TTS implementation to use')
@click.option('--voice-id', type=str, help='Voice ID for synthesis')
def speak(text, output, implementation, tts_model):
    """Convert text to speech"""
    config = ChrysalisConfig(
        tts_implementation=implementation,
        tts_model=tts_model
    )
    pipeline = ChrysalisPipeline(config)
    pipeline.synthesize_only(text, output_file=output)
    click.echo(f"Audio saved to: {output}")

@cli.command()
@click.option('--audio-file', type=str, help='Input audio file path')
@click.option('--use-mic', is_flag=True, help='Use microphone input')
@click.option('--stt-implementation', default='whisper_api',
              help='Speech-to-text implementation')
@click.option('--llm-model', default='ollama:llama2', help='LLM model to use')
@click.option('--tts-implementation', default='elevenlabs',
              help='Text-to-speech implementation')
@click.option('--output', type=str, default='output.mp3', help='Output audio file')
@click.option('--voice-id', type=str, help='Voice ID for synthesis')
def pipeline(audio_file, use_mic, stt_implementation, llm_model, 
            tts_implementation, output, tts_model):
    """Run full pipeline: STT -> LLM -> TTS"""
    config = ChrysalisConfig(
        stt_implementation=stt_implementation,
        llm_model=llm_model,
        tts_implementation=tts_implementation,
        tts_model=tts_model
    )
    
    pipeline = ChrysalisPipeline(config)
    results = pipeline.run(
        audio_file=audio_file,
        use_mic=use_mic,
        output_file=output
    )
    
    click.echo(f"Transcription: {results['transcription']}")
    click.echo(f"Response: {results['llm_response']}")
    click.echo(f"Audio saved to: {results['output_file']}")

if __name__ == '__main__':
    cli() 