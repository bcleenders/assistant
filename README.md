# Home assistant

Simple voice-controlled home assistant, to control Philips Hue lights via voice commands.

To run:

```bash
conda create -n assistant python==3.10
conda activate assistant
pip install -r requirements.txt

python assistant.py
```

The assistant will connect to a local Philips Hue and list all rooms and lights.

The assistant listens with the microphone and uses Silero to detect "blocks" of voice activity.

It streams these blocks to OpenAI Whisper (which runs locally) to transcribe the audio.

The transcribed texts are matched with SentenceTransformer to the most similar command that turns a light on or off (or discards it if it is not similar to any known command).

