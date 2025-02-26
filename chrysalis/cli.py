import click
from .pipeline import VoiceAssistantPipeline, PipelineConfig

@click.group()
def cli():
    """Voice Assistant CLI"""
    pass

@cli.command()
@click.option('--audio-file', type=str, help='Input audio file path')
@click.option('--use-mic', is_flag=True, help='Use microphone input')
@click.option('--implementation', type=str, default='whisper_api', 
              help='STT implementation to use')
def transcribe(audio_file, use_mic, implementation):
    """Transcribe speech to text"""
    config = PipelineConfig(stt_implementation=implementation)
    pipeline = VoiceAssistantPipeline(config)
    text = pipeline.transcribe_only(audio_file=audio_file, use_mic=use_mic)
    click.echo(f"Transcription: {text}")
    return text

@cli.command()
@click.argument('prompt')
@click.option('--model', default='ollama:llama2', help='LLM model to use')
def query(prompt, model):
    """Query LLM with text prompt"""
    config = PipelineConfig(llm_model=model)
    pipeline = VoiceAssistantPipeline(config)
    response = pipeline.query_only(prompt)
    click.echo(f"Response: {response}")
    return response

@cli.command()
@click.argument('text')
@click.option('--output', type=str, default='output.mp3', help='Output audio file')
@click.option('--implementation', type=str, default='elevenlabs',
              help='TTS implementation to use')
@click.option('--voice-id', type=str, help='Voice ID for synthesis')
def speak(text, output, implementation, voice_id):
    """Convert text to speech"""
    config = PipelineConfig(
        tts_implementation=implementation,
        voice_id=voice_id
    )
    pipeline = VoiceAssistantPipeline(config)
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
            tts_implementation, output, voice_id):
    """Run full pipeline: STT -> LLM -> TTS"""
    config = PipelineConfig(
        stt_implementation=stt_implementation,
        llm_model=llm_model,
        tts_implementation=tts_implementation,
        voice_id=voice_id
    )
    
    pipeline = VoiceAssistantPipeline(config)
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