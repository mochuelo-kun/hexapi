# Chrysalis

Chrysalis is a framework for exploring more natural and embodied forms of human-AI interaction through voice. Rather than following the traditional "voice assistant" model of rigid command-response patterns, Chrysalis aims to enable more fluid, contextual, and natural interactions.

## Vision

Chrysalis takes inspiration from how humans naturally converse and interact, acknowledging that meaningful interaction involves more than just exchanging information. Key aspects we're exploring:

- **Natural Conversation Flow**: Moving beyond the rigid query-response pattern to support more natural dialogue rhythms:
  - Variable response timing (sometimes listening longer before responding)
  - Non-sequential responses (referring back to earlier points)
  - Natural interjections and conversational markers
  - AI-initiated topics and questions
  - Comfortable silence and pauses

- **Physical Embodiment**: Exploring how AI interaction changes when embodied in physical forms:
  - Integration with physical objects (stuffed animals, robots)
  - Contextual awareness through sensors
  - Spatial and temporal awareness
  - Multi-modal interaction capabilities

- **Memory and Context**: Building up meaningful interaction history:
  - Remembering past conversations and contexts
  - Building internal models of people and places
  - Using RAG for grounded responses
  - Temporal awareness of interaction patterns

## Current Features

- Modular STT → LLM → TTS pipeline
- Swappable implementation options for each component:
  - Speech-to-Text: Local Whisper, OpenAI API
  - LLM: Local Ollama, OpenAI API, Local Server
  - Text-to-Speech: Local Piper, ElevenLabs API
- Simple CLI interface for testing components

## Installation

1. Clone the repository
`git clone https://github.com/yourusername/chrysalis.git`
`cd chrysalis`

2. Install with poetry
`poetry install`

3. Set up environment variables
`cp .env.example .env`

4. Edit .env with your API keys

5. Download Piper voice models (optional)
```bash
# Create models directory
mkdir -p ~/.piper-tts
cd ~/.piper-tts

# Download a model (e.g., English US Amy)
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.json
```

Note: MacOS users may have issues installing piper

TODO: figure out MacOS + piper problem (or remove until python piper fixes the problem?)


## Usage

Basic usage through CLI:

### Full pipeline (STT -> LLM -> TTS)

`poetry run chrysalis pipeline --use-mic`

### Individual components

`poetry run chrysalis transcribe --use-mic`

`poetry run chrysalis query "Hello, how are you?"`

`poetry run chrysalis speak "I'm doing well, thank you!"`

## Architecture

Chrysalis uses a modular architecture where each component (STT, LLM, TTS) is abstracted behind a common interface, allowing easy swapping of implementations. This enables experimentation with different models and approaches while maintaining a consistent API.

## Development Status

This project is in early experimental stages. Current focus areas:
- [ ] Implementing basic pipeline functionality
- [ ] Adding more model options for each component
- [ ] Developing natural conversation flow patterns
- [ ] Exploring embodiment options

## Contributing

We welcome contributions! Whether you're interested in:
- Adding new model implementations
- Improving conversation flow
- Exploring embodiment approaches
- Adding sensor integration
- Improving documentation

Please feel free to open issues or pull requests.

## License

[Your chosen license]