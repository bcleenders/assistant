import librosa
import pyaudio
from collections import deque
from faster_whisper import WhisperModel
import queue
import numpy as np
import torch
import datetime as dt
import time

SAMPLE_RATE = 16000
CHUNK_SIZE = int(SAMPLE_RATE / 10)
DEVICE = None

class MicrophoneStream:
    """Opens a recording stream as a generator yielding the audio chunks."""

    def __init__(self):
        # Create a thread-safe buffer of audio data
        self._buff = queue.Queue()
        self.closed = True

    def __enter__(self):
        self._audio_interface = pyaudio.PyAudio()
        self._audio_stream = self._audio_interface.open(
            # format=pyaudio.paInt16,
            format=pyaudio.paFloat32,
            # The API currently only supports 1-channel (mono) audio
            # https://goo.gl/z757pE
            channels=1,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK_SIZE,
            input_device_index=DEVICE,
            # Run the audio stream asynchronously to fill the buffer object.
            # This is necessary so that the input device's buffer doesn't
            # overflow while the calling thread makes network requests, etc.
            stream_callback=self._fill_buffer,
        )

        self.closed = False

        return self

    def __exit__(self, type, value, traceback):
        self._audio_stream.stop_stream()
        self._audio_stream.close()
        self.closed = True
        # Signal the generator to terminate so that the client's
        # streaming_recognize method will not block the process termination.
        self._buff.put(None)
        self._audio_interface.terminate()

    def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
        """Continuously collect data from the audio stream, into the buffer."""
        self._buff.put(in_data)
        return None, pyaudio.paContinue

    def generator(self):
        while not self.closed:
            # Use a blocking get() to ensure there's at least one chunk of
            # data, and stop iteration if the chunk is None, indicating the
            # end of the audio stream.
            chunk = self._buff.get()
            if chunk is None:
                return
            data = [chunk]

            # Now consume whatever other data's still buffered.
            while True:
                try:
                    chunk = self._buff.get(block=False)
                    if chunk is None:
                        return
                    data.append(chunk)
                except queue.Empty:
                    break

            ans = np.frombuffer(b"".join(data), dtype=np.float32)
            # yield uniform-sized chunks
            ans = np.split(ans, np.shape(ans)[0] / CHUNK_SIZE)
            for chunk in ans:
                yield chunk

def get_microphone_chunks(
    microphone_stream,
    vad_model,
    min_to_cumulate=5,  # 0.5 seconds
    max_to_cumulate=100,  # 10 seconds
    precumulate=5,
    max_to_visualize=100,
    speech_breaks=3 # 0.3 second break between words can occur
):
    # Store speech chunks
    cumulated = np.ndarray([0])
    # Store a few chunks back (up to $precumulate chunks)
    precumulated = deque(maxlen=precumulate)

    no_speech_cnt = 0

    with microphone_stream as stream:
        audio_generator = stream.generator()
        waveform = torch.zeros(max_to_visualize * CHUNK_SIZE)

        for chunk in audio_generator:
            # Is speech?
            speech_confidence = vad_model(torch.from_numpy(chunk.copy()), SAMPLE_RATE).item()
            is_speech = speech_confidence > 0.5

            # Print speech confidence with a histogram-like visualisation
            print(f'{speech_confidence:.3f} - [{("*" * int(speech_confidence * 50)):50s}]')
            
            # Are we seeing speech now?
            if is_speech or cumulated.size > 0:
                # cumulated.append(torch.from_numpy(chunk))
                cumulated = np.append(cumulated, chunk)
            else:
                # This is pre-speech, so pre-cumulate it into our fifo queue
                precumulated.append(chunk)

            # Measure silence (between potentially connected chunks)
            if is_speech:
                no_speech_cnt = 0
            else:
                no_speech_cnt += 1

            # Stop conditions
            stopped_speaking = (no_speech_cnt > speech_breaks) and not is_speech # the "not is_speech" is implied but added for clarity
            reached_min_length = (cumulated.size / CHUNK_SIZE) >= (min_to_cumulate + speech_breaks)
            exceeded_max_length = cumulated.size / CHUNK_SIZE > max_to_cumulate
            if (stopped_speaking and reached_min_length) or exceeded_max_length:
                waveform = np.append(np.array(precumulated), cumulated)
                yield waveform
                cumulated = np.ndarray([0])
                precumulated = deque(maxlen=precumulate)

def transcribe(waveform, stt_model):
    segments, info = stt_model.transcribe(waveform, beam_size=5)
    s = "".join([s.text for s in segments])
    return s.strip()

def get_microphone_transcription(audio_chunk_stream, stt_model):
    for waveform in audio_chunk_stream:
        transcription = transcribe(waveform, stt_model)
        yield transcription

def setup_asr():
    model_size = "base.en"
    whisper_model = WhisperModel(model_size, device="cpu", compute_type="int8")

    silero_model, utils = torch.hub.load(
                              repo_or_dir='snakers4/silero-vad',
                              model='silero_vad',
                              force_reload=False)

    return (whisper_model, silero_model)

# HUE
import requests
from huesdk import Hue, Discover
from pathlib import Path
import json

def get_hue_bridge_ip():
    # discovered = requests.get('https://discovery.meethue.com/')
    # discovered.json()
    discover = Discover()
    discovered = json.loads(discover.find_hue_bridge_mdns(timeout=5))

    if len(discovered) == 0:
        print("No Hue found on this network :(")
    else:
        hue_bridge_ip = discovered[0]['internalipaddress']
        print(f"Found Hue at: {hue_bridge_ip}")

        return hue_bridge_ip

def get_hue_user(hue_bridge_ip):
    # Ignore warnings from urllib3 about insecure http connection
    import urllib3
    urllib3.disable_warnings()

    username_file = Path("username.txt")
    username = ""

    if username_file.is_file():
        with open(username_file, 'r') as f:
            username = f.read()
            f.close()
            print(f"Using cached username (from username.txt)")
    else:
        print("Did not find cached username - connect to bridge instead")

    while username == "":
        input("Press the 'Connect' button on Hue, then press Enter to continue.")
        username = Hue.connect(bridge_ip=hue_bridge_ip)

    with open(username_file, 'w') as f:
        f.write(username)
        f.close()
    return username

def setup_hue():
    hue_bridge_ip = get_hue_bridge_ip()
    username = get_hue_user(hue_bridge_ip)
    hue = Hue(bridge_ip=hue_bridge_ip, username=username)

    return hue

# Hides output from functions
from contextlib import contextmanager
import os
import sys
@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout

def get_hue_commands(hue):
    commands = []

    with suppress_stdout():
        lights = hue.get_lights()

    rooms = hue.get_groups()
    targets = lights + rooms

    for target in targets:
        # Somewhat convoluted - we need a copy of target to keep it in scope
        createOnSwitch = lambda t: lambda: t.on()
        createOffSwitch = lambda t: lambda: t.off()
        
        commands.append({
            'phrase': f"Turn the {target.name} light on",
            'action': createOnSwitch(target)
        })
        commands.append({
            'phrase': f"Turn the {target.name} light off",
            'action': createOffSwitch(target)
        })

    commands.append({
        'phrase': 'Stop listening',
        'action': None
        })

    return commands

def get_command_stream(transcription_stream, commands):
    from sentence_transformers import SentenceTransformer, util
    import torch

    sentencetransformer = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    #  These are known phrases that we can link with commands
    phrases = [command['phrase'] for command in commands]
    embeddings = sentencetransformer.encode(phrases)

    print(f"I know {len(phrases)} commands.")
    
    for transcription in transcription_stream:
        # Find most similar command available
        needle_embedding = sentencetransformer.encode(transcription)
        # Matrix with co-similarities, 1*N, where 1=#needles, N=#embeddings
        # Values between -1 (opposite) and 1 (equal)
        cos_sim = util.cos_sim(needle_embedding, embeddings)[0]
        best_match_index = torch.argmax(cos_sim)
        
        yield (
            transcription,
            cos_sim[best_match_index],
            commands[best_match_index]['phrase'],
            commands[best_match_index]['action']
        )

def main():
    (stt_model, vad_model) = setup_asr()
    
    hue = setup_hue()
    hue_commands = get_hue_commands(hue)

    # Audio stream
    microphone_stream = MicrophoneStream()
    # Audio stream, split into chunks representing sentences
    audio_chunk_stream = get_microphone_chunks(microphone_stream, vad_model)
    # Stream of transcriptions, generated from audio chunks
    transcription_stream = get_microphone_transcription(audio_chunk_stream, stt_model)
    # Stream of commands, as parsed from the transcriptions
    command_stream = get_command_stream(transcription_stream, hue_commands)

    print("READY!")

    for (transcription, cos_sim, cmd_phrase, cmd_action) in command_stream:
        print(f"I think you said:      {transcription}")
        print(f"  which is similar to: {cmd_phrase}")
        print(f"  with likelyhood:     {cos_sim} (between -1 and 1)")
        if cos_sim > 0.75:
            if cmd_phrase == 'Stop listening':
                return
            cmd_action()
            print("  which is a good match, so I will executed it.")
        else:
            print("  which is not super similar, so I will ignore this. It is likely not a command.")

        time.sleep(1)

if __name__ == "__main__":
    main()
