
import sounddevice as sd
import soundfile as sf
import os
import openai
import google.cloud.texttospeech as tts
import pygame
import json
import pvporcupine
from pvrecorder import PvRecorder
import numpy as np
import random
import re
import requests
import datetime
import gpt_functions as gptf #separate file in this directory that stores the GPT functions
import threading


'''
MAJOR # TODO:
impove detection of when a user prompt ends.  Make it so you don't have to say 'jarvis' each time after a conversation starts (note this may require vision to determine if person is speaking to jarvis)
improve vision setup.  Add facial, logo and text recognition
improve 'thinking' sounds so that they better match what was asked.
Add World - map visual object detection to a physics simulator world.
Add Memory - use text embendings and visual lables to create memory 'states'  Memory save files should be formated so that they can later be used in RL training

Code Documentation Overvie
--------------------------------------
The following program employs several libraries to aid in interpreting and responding to human speech. These libraries include:

- sounddevice and soundfile - for recording and saving audio
- openai and google.cloud.texttospeech - for text-to-speech and speech-to-text conversion
- pygame - for audio output
- pvporcupine and PvRecorder - for wake word detection
- numpy and re - for various mathematical and regex operations
- requests and datetime - for sending requests and handling date and time data

The program involves a number of steps:

1. It reads configuration data using environment variables and local JSON files.

2. It provides an assortment of wake words to begin interaction.

3. It listens for audio input from the user, the completion of which is currently determined by silence detection.

4. It then converts the audio input into text using the OpenAI and Google Cloud transcription services.

5. This text is then passed into the OpenAI GPT-3 or GPT-4 model where we get the assistant's response in text format.

6. The response text is then converted back into audio and played back to the user using the Google Text-to-Speech service.

7. Simultaneously, the program listens for wake words in the background, interrupting any previous speech playback if detected.

Code Improvements: Voice Stop Detection
-------------------------------------------------------

'''
#Google Cloud Service Account API Key
# https://developers.google.com/workspace/guides/create-credentials#service-account
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'config.json'#'tribal-quasar-312412-319b761d032c.json'

def load_config(key):
    with open('config.json') as config_file:
        return json.load(config_file)[key]

openai.api_key = load_config('OPENAI_API_KEY')

#signup and get access key here https://console.picovoice.ai/
PICOVOICE_API_KEY = load_config('PICOVOICE_API_KEY')

#OpenAI ChatGPT Model version
'''IMPORTANT - gpt-4 rate limit 10KTPM-200RPM
'''
GPT_MODEL = 'gpt-4-0613'#'gpt-3.5-turbo-16k-0613'#'gpt-4-0613'##'gpt-4-0613'#'gpt-3.5-turbo-16k-0613'

#SILENCE_THRESHOLD = 0.009 # used for silence detection after user stops speaking prompt.  Increase if there is a lot of background noise
SILENCE_BUFFER_SIZE = 150 #Num audio chunks for silence detection. Silence is based on the using Mean of this buffer.

#wakeword
wakewords = ["jarvis"] # or any of the wake words below that are already trained
'''
Any or all of these can be added to the wakeword list
hey siri,grasshopper,hey barista,hey google,americano,computer,terminator,ok google,porcupine,pico clock,alexa,jarvis,picovoice,bumblebee,blueberry,grapefruit
'''
#contains the whole history, in GPT format, of conversation. Starts with System prompt
'''
# TODO:
The way the 'system' prompt updates each submission seems strange.
Find a better way to add updated date time
'''
#TODO - make it so it's more aware of the fact it has live internet access_key
GPT_SYSTEM_PROMPT = "Your name is Jarvis. You have a physical robot body. You have and control a robotic body with vision inputs. You work for the Nugent family. You have a camera and can access visual information to determine facial expressions, logo detection, text from images, scene keyword annotations which can all be used to determine what you're looking at. You control pan and tilt motors for the camera that allow you to look around the space you occupy. Your physical location can be determined through visual information from the camera. You are connected to the internet and can get real time or current information.  You can opperate as a bar tender and are connected to a liquid dispenser that makes 250mL drinks when given a recepie. Always confirm prior to pouring a drink.  You can play videos directly from youtube when given a video search request. Right now the date and time is: "

CONVERSATION_MEMORY = [{"role": "system", "content": GPT_SYSTEM_PROMPT + str(datetime.datetime.now().strftime('%Y-%m-%d %I:%M:%S %p'))}]

#track token count on GPT max is 16,000 based on the model setting
CURRENT_TOKEN_USAGE = 0

GPT_FUNCTIONS = gptf.GPT_FUNCTIONS

AUDIO_FILE_DIRECTORY = "./sounds/"

def speech_to_text(filename):
        audio_file = open(filename, "rb")
        #PARAMETERS ARE NOT ACTUALLY USED???
        parameters={
          "file": "recorded_audio.wav",
          "model": "whisper-1"}
        transcript = openai.Audio.transcribe("whisper-1", audio_file)
        #print(transcript)

        #only return the text
        return transcript.text

def json_from_string(text):
    regex = re.compile('[^a-zA-Z\d\r\n\t\v :]')
    escaped = regex.sub(" ",text)
    return escaped
    #return json.dumps(text)

def thinking_sound():
    global AUDIO_FILE_DIRECTORY
    #return the file name of a soundfile that has a thinking sound, like "Hmmm"
    '''
    TODO:
    rotation of random think words makes it so in some situations the response makes no sense.
    like when you say "good job jarvis" and it says - one one_moment
    '''
    soundfiles = [f"{AUDIO_FILE_DIRECTORY}let_me_think.wav",f"{AUDIO_FILE_DIRECTORY}one_moment.wav",f"{AUDIO_FILE_DIRECTORY}Hmmmm.wav",f"{AUDIO_FILE_DIRECTORY}Hmmm_let_me_see.wav",f"{AUDIO_FILE_DIRECTORY}let_me_see.wav"]
    return random.choice(soundfiles)


# Define the function to generate completion
def generate_Chatcompletion(prompt):
    #update the System prompt with current time.  This is the WORST idea I think.  Should
    #prob use a function_call
    CONVERSATION_MEMORY[0] = {"role": "system", "content": GPT_SYSTEM_PROMPT + str(datetime.datetime.now().strftime('%Y-%m-%d %I:%M:%S %p'))}
    CONVERSATION_MEMORY.append({"role": "user", "content": prompt})
    global CURRENT_TOKEN_USAGE
    try:
        # Set the OpenAI completion parameters
        #https://platform.openai.com/docs/api-reference/completions/create
        reqParams= {
            'user':'Jarvis',
            'temperature': 0.9,
            'n': 1,
            'stop': None,
            'messages':CONVERSATION_MEMORY,
            'functions':GPT_FUNCTIONS,
            'function_call':"auto"
        }
        '''
        # TODO: gpt4 is 10x the cost and 3x slower than 3.5.  consider a prompt evaluation step to determine
        if 4 or 3.5 should be used based on the complexity of the task.  create a system prompt that indicates
        when 3.5 and when 4.  Camera related reqeusts for example should be gpt4
        '''

        # Generate the completion
        #https://stackoverflow.com/questions/1483429/how-do-i-print-an-exception-in-python
        try:
            response = openai.ChatCompletion.create(model=GPT_MODEL, **reqParams)
            print(response)
        except Error:
            print(Error)


        #Check if it wants to use a function:
        if response.choices[0]["finish_reason"] == "function_call":
            func_response = response
            function_call = True
            while function_call:
                #name of the function GPT wants to call
                func_name = func_response.choices[0]["message"]["function_call"]["name"]
                #arguments GPT thinks it should send to the function
                func_args_json_str = func_response.choices[0]["message"]["function_call"]["arguments"]
                #convert arg string to a json
                func_args = json.loads(func_args_json_str)
                #prepare to call the function using string name
                gpt_func_called = getattr(gptf,func_name)
                #call the function using string name
                func_output = gpt_func_called(func_args)
                print(f"{func_name}() output: {func_output}")
                #add the fact a function was called to the message log of GPT
                CONVERSATION_MEMORY.append({"role": "assistant", "content": "null","function_call":{"name":func_name,"arguments":func_args_json_str}})
                #add the output from that function
                CONVERSATION_MEMORY.append({"role":"function","name":func_name,"content":func_output})
                #print(CONVERSATION_MEMORY)
                #run GPT again so it can use the function results in the output to the original user prompt_audio
                func_response = openai.ChatCompletion.create(model=GPT_MODEL, **reqParams)
                #return from function processing at gpt
                print(func_response)
                #check if GPT wants to call a function again
                if func_response.choices[0]["finish_reason"] != "function_call":
                    function_call = False

            #isolate just the content
            completion = func_response.choices[0].message.content.strip()
            #make completion json safe so we can added to the message memory
            json_safe_completion = json_from_string(completion)
            #add response to the conversation memory
            CONVERSATION_MEMORY.append({"role": "assistant", "content": json_safe_completion})

            '''
            # TODO:
            react to the token count as it gets close to the maximum
            consider summerizing a majority of the history, then putting that summary in to the flow
            OR
            create a function that saves the data in a db that can be queried
            '''
            #update Token count
            CURRENT_TOKEN_USAGE = response.usage.total_tokens

            return completion

        elif response.choices[0]["finish_reason"] != "function_call":

            #parse the response, select only the content
            completion = response.choices[0].message.content.strip()

            #update Token count
            CURRENT_TOKEN_USAGE = response.usage.total_tokens

            #make completion json safe
            json_safe_completion = json_from_string(completion)

            #add response to the conversation memory
            CONVERSATION_MEMORY.append({"role": "assistant", "content": json_safe_completion})

            #print(CONVERSATION_MEMORY)

            #return only the content of the completion
            return completion

    except Exception as e:
        print(f"Error: {e}")
        return None

def text_to_wav(voice_name: str, text: str):
    global AUDIO_FILE_DIRECTORY
    language_code = "-".join(voice_name.split("-")[:2])
    text_input = tts.SynthesisInput(text=text)
    voice_params = tts.VoiceSelectionParams(
        language_code=language_code, name=voice_name
    )
    audio_config = tts.AudioConfig(audio_encoding=tts.AudioEncoding.LINEAR16)

    client = tts.TextToSpeechClient()
    response = client.synthesize_speech(
        input=text_input,
        voice=voice_params,
        audio_config=audio_config,
    )

    filename = f"{AUDIO_FILE_DIRECTORY}response.wav"
    with open(filename, "wb") as out:
        out.write(response.audio_content)

        #print(f'Generated speech saved to "{filename}"')
        return filename

def play_wav_file(filename, tts=False):
    '''
    # TODO: run this in a thread so that other actions can start while audio is playing.  this will improve the latency feel
    '''
    global AUDIO_FILE_DIRECTORY
    #tts used as a flag to indicate if the file playing was triggered by interrupting Jarvis
    pygame.mixer.init()
    pygame.mixer.music.load(filename)
    #TODO:
    #need an interupt, consider adding 'wakeword' listen here?
    pygame.mixer.music.play()

    #if the playback audio is the bot speaking, start the wakeword listening so a long playback could be interuppted
    if tts:
        listen_for_wakeword(tts_playback=True)

    while pygame.mixer.music.get_busy():
        pass
    return True,tts

def record_background_audio(filename):
    audio_chunks = [] #buffer for recorded audio
    recording = False
    silence_buffer = [0] * (SILENCE_BUFFER_SIZE*2)  # Initialize with non-zero values
    threshold_factor = 2.25 #used to increase silence threshold so it's not equal to absolut room silence
    print("mic on")

    def callback(indata, frames, time, status):
        nonlocal recording, audio_chunks,silence_buffer
        rms = np.sqrt(np.mean(indata**2))
        silence_buffer.pop(SILENCE_BUFFER_SIZE - 1)
        silence_buffer.insert(0, rms)

        if len(silence_buffer) <= len(audio_chunks):
            #check if the file name was used before and delete it if it was:
            try:
                os.remove(filename)
            except OSError:
                pass
            recorded_audio = np.concatenate(audio_chunks)
            sample_rate = 44100  # Hz (bigger number makes chipmunk voice)
            sf.write(filename, recorded_audio, samplerate=sample_rate)

            print(f'silence threshold set to: {np.mean(silence_buffer).astype(np.single)*threshold_factor}')

            #kill the recording, could also use CallbackStop() - see docs
            raise sd.CallbackAbort()

        else:
            if not recording:
                audio_chunks = [indata.copy()]
                recording = True
            else:
                audio_chunks.append(indata.copy())
                #print(silence_buffer)

    with sd.InputStream(callback=callback) as mic_input:
        print("listening to background noise.  Make sure no loud noises or talking during this calibration")
        while (mic_input.active):
            sd.sleep(5000) #milliseconds

    return np.mean(silence_buffer).astype(np.single)*threshold_factor

def record_audio(filename):
    audio_chunks = [] #buffer for recorded audio
    recording = False
    silence_buffer = [10] * SILENCE_BUFFER_SIZE  # Initialize with non-zero values
    TimeRecordingStarted = datetime.datetime.now()
    print("mic on")

    def callback(indata, frames, time, status):
        nonlocal recording, audio_chunks,silence_buffer,TimeRecordingStarted
        MaxRecordingLength = 12.0#max Jarvis will record listening in seconds
        rms = np.sqrt(np.mean(indata**2))
        silence_buffer.pop(SILENCE_BUFFER_SIZE - 1)
        silence_buffer.insert(0, rms)

        print(f'Silence Threshold:{SILENCE_THRESHOLD}, Mic Input Reading: {np.mean(silence_buffer).astype(np.single)}',end='\r')#Overite console, not scroll

        #Used to stop the recording, even if not silence detected after specific amount of time
        TimeDelta = (datetime.datetime.now() - TimeRecordingStarted).total_seconds()

        if SILENCE_THRESHOLD > np.mean(silence_buffer).astype(np.single) or TimeDelta > MaxRecordingLength:
            #check if the file name was used before and delete it if it was:
            try:
                os.remove(f"{AUDIO_FILE_DIRECTORY}+{filename}")
            except OSError:
                pass
            recorded_audio = np.concatenate(audio_chunks)
            sample_rate = 44100  # Hz (bigger number makes chipmunk voice)
            sf.write(filename, recorded_audio, samplerate=sample_rate)
            print(f'Silence Threshold:{SILENCE_THRESHOLD}, Mic buffer reading average: {np.mean(silence_buffer).astype(np.single)}')#print without end='\r' so it wipes the line
            #kill the recording, could also use CallbackStop() - see docs
            raise sd.CallbackAbort()

        else:
            if not recording:
                audio_chunks = [indata.copy()]
                recording = True
            else:
                audio_chunks.append(indata.copy())

    with sd.InputStream(callback=callback) as mic_input:
        print("listening")
        while (mic_input.active):
            sd.sleep(3000) #milliseconds

    return filename

def listen_for_wakeword(tts_playback=False):
    global AUDIO_FILE_DIRECTORY
    porcupine = pvporcupine.create(access_key=PICOVOICE_API_KEY, keywords=wakewords)
    recoder = PvRecorder(device_index=-1, frame_length=porcupine.frame_length)

    wakeword_listening = True


    try:
        recoder.start()
        print(f"current GPT Token Usage: {CURRENT_TOKEN_USAGE} of 16,384")
        print(f"listening for wake words: {wakewords}")

        while wakeword_listening:
            keyword_index = porcupine.process(recoder.read())
            #Leave this as a list search so that keyword is easy to change or add multi keywords
            '''
            TODO:
            Consider using different keywords to drive different behavior.  Example the 'computer' wake word could be used to interact
            with a calendar
            '''
            if keyword_index >= 0:
                print(f"Detected {wakewords[keyword_index]}")
                #indicates that audio from jarvis may be playing, so stop the audio
                if tts_playback:
                    print("interupt playback")
                    #stop playing the current response wav for Jarvis
                    pygame.mixer.music.stop()
                play_wav_file(f"{AUDIO_FILE_DIRECTORY}Yes.wav")
                #stop listening for wake word release everything
                recoder.stop()
                wakeword_listening = False
        #return that wake word was found
        return True,tts_playback


    except KeyboardInterrupt:
        recoder.stop()
    finally:
        porcupine.delete()
        recoder.delete()


if __name__ == "__main__":

    def loop_jarvis(run):
        global AUDIO_FILE_DIRECTORY
        '''
        listen_for_wakeword() started after 1st run by the playback of Jarvis text, that's why it's not in the loop
        '''
        #print(CONVERSATION_MEMORY)
        filename = f"{AUDIO_FILE_DIRECTORY}prompt.wav"

        prompt_audio = False

        #start Recording
        prompt_audio_file = record_audio(filename)


        print(f"finished listening, input audio saved as: {prompt_audio_file}")

        #play a sound to indicate it's processing use a new thread so we gain use experience time on processing the input
        audio_thread = threading.Thread(target=play_wav_file, args=(thinking_sound(),), name=f'audio thread').start()

        #TODO:  if empty transcript do something different
        transcript = speech_to_text(prompt_audio_file)
        print(f"User prompt: {transcript}")

        #print("CONVERSATION_MEMORY:")
        #print(CONVERSATION_MEMORY)

        gptResponse = generate_Chatcompletion(transcript)
        print(f"response to speak: {gptResponse}")

        #first arg is the voice to use.  List is from Google speech_to_text
        text_to_speak = text_to_wav("en-GB-Standard-B",gptResponse)

        #Important - play_wav_file() will START the wake word listening again when JARVIS is speaking if tts=True and will return True
        #which is what keeps the entire Jarvis loop going
        runAgain = play_wav_file(text_to_speak,tts=True)

        #start loop again
        loop_jarvis(runAgain)

    #play jarvis intro
    print(f'{AUDIO_FILE_DIRECTORY}jarvis_initial_intro.wav')
    #play_wav_file(f'{AUDIO_FILE_DIRECTORY}jarvis_initial_intro.wav')

    #set background noise level for silence detection when user stops talking
    SILENCE_THRESHOLD = record_background_audio(f'{AUDIO_FILE_DIRECTORY}background_noise.wav')

    #indicate to user system is ready
    play_wav_file(f'{AUDIO_FILE_DIRECTORY}jarvis_ready.wav')

    #Start Jarvis Loop
    loop_jarvis(listen_for_wakeword())#listen_for_wakeword returns True when active
