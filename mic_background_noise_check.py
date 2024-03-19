import sounddevice as sd
import soundfile as sf
import numpy as np
import os

SILENCE_BUFFER_SIZE = 150

def record_background_audio(filename):
    audio_chunks = [] #buffer for recorded audio
    recording = False
    silence_buffer = [0] * (SILENCE_BUFFER_SIZE*5)  # Initialize with non-zero values
    print("mic on")

    def callback(indata, frames, time, status):
        nonlocal recording, audio_chunks,silence_buffer
        rms = np.sqrt(np.mean(indata**2))
        silence_buffer.pop(SILENCE_BUFFER_SIZE - 1)
        silence_buffer.insert(0, rms)

        #print(f'Silence Threshold:{SILENCE_THRESHOLD}, Mic Input Reading: {np.mean(silence_buffer).astype(np.single)}')

        if len(silence_buffer) <= len(audio_chunks):
            #check if the file name was used before and delete it if it was:
            try:
                os.remove(filename)
            except OSError:
                pass
            recorded_audio = np.concatenate(audio_chunks)
            sample_rate = 44100  # Hz (bigger number makes chipmunk voice)
            sf.write(filename, recorded_audio, samplerate=sample_rate)

            #print(f'threshold set to: {np.mean(silence_buffer).astype(np.single)*1.1}')

            #kill the recording, could also use CallbackStop() - see docs
            raise sd.CallbackAbort()

        else:
            if not recording:
                audio_chunks = [indata.copy()]
                recording = True
            else:
                audio_chunks.append(indata.copy())

    with sd.InputStream(callback=callback) as mic_input:
        print("listening to background noise for 10 seconds.  Make sure no loud noises or talking during this calibration")
        while (mic_input.active):
            sd.sleep(5000) #milliseconds

    return np.mean(silence_buffer).astype(np.single)*1.1

SILENCE_THRESHOLD = record_background_audio('background_noise.wav')

print(SILENCE_THRESHOLD)
