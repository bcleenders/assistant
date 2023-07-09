import librosa
import pyaudio
from collections import deque
from faster_whisper import WhisperModel
import queue
import numpy as np
import torch

class MicrophoneStream:
    """Opens a recording stream as a generator yielding the audio chunks."""

    def __init__(self, device=None, rate=16000, chunk=1600):
        self._rate = rate
        self._chunk = chunk
        self._device = device

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
            rate=self._rate,
            input=True,
            frames_per_buffer=self._chunk,
            input_device_index=self._device,
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

            # for chunk in data:
            #     yield chunk
            ans = np.frombuffer(b"".join(data), dtype=np.float32)
            # ans = np.fromstring(b"".join(data), dtype=np.float32)
            # yield uniform-sized chunks
            ans = np.split(ans, np.shape(ans)[0] / self._chunk)
            # Resample the audio
            for chunk in ans:
                yield chunk
                # yield librosa.core.resample(chunk, self._rate, 16000)

def int2float(sound):
    abs_max = np.abs(sound).max()
    sound = sound.astype('float32')
    if abs_max > 0:
        sound *= 1/32768
    sound = sound.squeeze()  # depends on the use case
    return sound

def get_microphone_chunks(
    vad_model,
    min_to_cumulate=5,  # 0.5 seconds
    max_to_cumulate=100,  # 10 seconds
    precumulate=5,
    max_to_visualize=100,
):
    precumulated = deque(maxlen=precumulate)

    with MicrophoneStream() as stream:
        audio_generator = stream.generator()
        chunk_length = stream._chunk
        waveform = torch.zeros(max_to_visualize * chunk_length)

        for chunk in audio_generator:
            # Is speech?
            # audio_chunk = stream.read(num_samples)
            # audio_int16 = np.frombuffer(chunk, np.int16);
            # audio_float32 = int2float(audio_int16)
            print(chunk.shape)
            speech_confidence = vad_model(torch.from_numpy(chunk), 16000).item()

            # Print speech confidence with a histogram-like visualisation
            print(f'{speech_confidence:.3f} - [{("*" * int(speech_confidence * 50)):50s}]')
            yield [1, 1]

            # Cumulate speech
            # if is_speech or cumulated:
            #     cumulated.append(chunk)
            # else:
            #     precumulated.append(chunk)

            # if (not is_speech and len(cumulated) >= min_to_cumulate) or (
            #     len(cumulated) > max_to_cumulate
            # ):
            #     waveform = torch.cat(list(precumulated) + cumulated, -1)
            #     yield (waveform * stream._rate, stream._rate)
            #     cumulated = []
            #     precumulated = deque(maxlen=precumulate)

def transcribe(waveform, model):
    return "foo"

def get_microphone_transcription(stt_model, vad_model):
    for (waveform, sample_rate) in get_microphone_chunks(vad_model):
        # waveform = torchaudio.transforms.Resample(
        #     orig_freq=sample_rate, new_freq=16000
        # )(waveform.reshape(1, -1))
        transcription = transcribe(waveform, stt_model)
        yield transcription

def setup_asr():
    model_size = "base.en"
    whisper_model = WhisperModel(model_size, device="cpu", compute_type="int8")

    silero_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                              model='silero_vad',
                              force_reload=True)

    return (whisper_model, silero_model)

def main():
    # logger = logging.getLogger(__name__)
    # logger.setLevel(logging.INFO)
    # task, generator, models, sp, tgt_dict = setup_asr(args, logger)
    (stt_model, vad_model) = setup_asr()

    print("READY!")
    # for transcription in get_microphone_transcription(args, task, generator, models, sp, tgt_dict):
    for transcription in get_microphone_transcription(stt_model, vad_model):
        if 1 == 2:
            print(
                transcription
                # "{}: {}".format(
                #     dt.datetime.now().strftime("%H:%M:%S"), transcription[0][0]
                # )
            )

if __name__ == "__main__":
    main()
