
import json
import requests
import matplotlib.pyplot as plt
import matplotlib.image as img
import pygame
import openai
from bs4 import BeautifulSoup
from bs4 import BeautifulSoup
from bs4.element import Comment
import urllib.request
import re
import datetime
import time
import random
from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
import threading
import cv2 as cv #- pip3 install opencv-python
from google.cloud import vision #pip3 install --upgrade google-cloud-vision --user
import os
import ssl
import serial

#have to add this because of some MacOS issues.
ssl._create_default_https_context = ssl._create_unverified_context

def load_config(key):
    with open('config.json') as config_file:
        return json.load(config_file)[key]

#Google Cloud Service Account API Key
# https://developers.google.com/workspace/guides/create-credentials#service-account
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'config.json' #Service Account json exported file
#all other API keys were put in the config.json output by Google Service Account
openai.api_key = load_config('OPENAI_API_KEY')
YOUTUBE_DATA_API_KEY = load_config('YOUTUBE_DATA_API_KEY')
GOOGLE_SEARCH_API_KEY = load_config('GOOGLE_SEARCH_API_KEY')
#locations used for sounds and images
AUDIO_FILE_DIRECTORY = "./sounds/"
IMAGE_FILE_DIRECTORY = "./images/"

'''
,
"imageAnalysis": {
    "description": "imageAnalysis is the list of availible image analysis that can be run.  More than one can be sent in a single request",
    "type":"list",
    "analysis":["label_detection","object_localization","text_detection","logo_detection","face_detection"],
}
'''

GPT_FUNCTIONS = [{
        "name": "gpt_google_search",
		"description": "An internet search using google that returns a snippet of text to answer questions about current events and provide access to real-time information",
		"parameters": {
			"type": "object",
			"properties": {
				"query": {
					"type": "string",
					"description": "accepts a string input to search the internet"
				}
			},
			"required": ["query"]}},
    {"name": "gpt_youtube",
    		"description": "Searches then plays a video related to a search or topic input.  The playback is automatically handled by the users browser",
    		"parameters": {
    			"type": "object",
    			"properties": {
    				"query": {
    					"type": "string",
    					"description": "accepts a string input to search the youtube for a relavant video"
    				}
    			},
    			"required": ["query"]
    		}},
    {"name": "gpt_vision",
    "description": "Access connected camera. Returns a json of up to items: detected objects, object annotation, Face detection, text recognized, logo detection. This information can be used to summarize what is in view, detect people, read text that is in view and identify logos in view. send the key and get for each analysis to be performed. returns the result from each analysis on the current camera view. Multiple analysis requests types can be sent in a single request",
    "parameters": {
    			"type": "object",
    			"properties": {
                    "label_detection":{
                    "type": "boolean",
                    "description":"returns a description for the image in view based on image keyword embedings."
                    },
                    "object_localization":{
                    "type": "boolean",
                    "description":"returns a list of detected objects and their X,Y loctions for the image in view."
                    },
                    "text_detection":{
                    "type": "boolean",
                    "description":"returns a string of detected text for the image in view, as well as X,Y locations of words and sentences."
                    },
                    "logo_detection":{
                    "type": "boolean",
                    "description":"returns a json of all logos detected for the image in view and X,Y locations of where the logos are."
                    },
                    "face_detection":{
                    "type": "boolean",
                    "description":"returns a json of all faces detected and X,Y locations of where the faces are in the image in view. Facial sentiment and expression is also detected and returned for each face"
                    }
    			},
    		}
    	},
    {"name": "gpt_get_availible_ingredients",
		"description": "Get database information regarding ingredients in the drink mixer. It will return a list of ingredients.  Calling this function is the only way to determine what ingredients are available",
		"parameters": {
			"type": "object",
			"properties": {
				"query": {
					"type": "string",
					"description": "A SQL query of all ingredients"
				}
			},
			"required": ["query"]
		}},
    {"name": "gpt_dispense_mixed_drink",
	   "description": "Important! before calling this function, confirm with the user that they want you to dispense the drink. This will dispense a 250mL liquid drink by inputting recepie consisting of the ingredients list and the amounts of that ingredient to be added to the drink in mL",
	      "parameters": {
          "type":"object",
		"properties": {
			"ingredients": {
				"description": "ingredient name and the mL amount of that ingredient to be dispensed",
				"ingredient": {
					"ingredient": "mL measurment",
					"ingredient": "mL measurment"
				}
			}

		}
	},
	       "required": ["ingredients"]
},    {"name": "gpt_camera_motor_control",
    "description": "control pan and tilt motors for camera, input two positive numbers one for pan one for tilt that represent degrees to move the camera where tilt 75 and pan 75 is center, pan 180 is all the way left, pan 0 is all the way right, tilt 0 is straight up, tilt 180 is straight down.",
    "parameters": {
    			"type": "object",
    			"properties": {
    				"pan": {
    					"type": "string",
                        "description":"A number between 0 and 100 that represents degrees of pan where 75 is center",
                        "unit":{"enum":"pan"}
    				},
                    "tilt": {
    					"type": "string",
                        "description":"A number between 0 and 100 that represents degrees of tilt where 75 is center",
                        "unit":{"enum":"tilt"}
    				}
    			},
    			"required": ["pan","tilt"]
    		}
    	},
        {"name": "gpt_hand_motor_control",
            "description": "control each finger and the thumb of the robot hand, input positive numbers from 0 to 100 that sets the extendness of fingers.  0 is all the way closed for any finger or thumb. 100 will fully extend any finger or thumb.",
            "parameters": {
            			"type": "object",
            			"properties": {
            				"pinky": {
            					"type": "string",
                                "description":"A number that represents degrees of finger extension. 0 is fully closed. 100 is fully extended",
                                "unit":{"enum":"pinky"}
            				},"ring": {
            					"type": "string",
                                "description":"A number that represents degrees of finger extension. 0 is fully closed. 100 is fully extended",
                                "unit":{"enum":"ring"}
            				},
                            "middle": {
            					"type": "string",
                                "description":"A number that represents degrees of finger extension. 0 is fully closed. 100 is fully extended",
                                "unit":{"enum":"middle"}
            				},
                            "index": {
            					"type": "string",
                                "description":"A number that represents degrees of finger extension. 0 is fully closed. 100 is fully extended",
                                "unit":{"enum":"index"}
            				},
                            "thumb": {
            					"type": "string",
                                "description":"A number that represents degrees of finger extension. 0 is fully closed. 100 is fully extended",
                                "unit":{"enum":"thumb"}
            				},
                            "wrist": {
            					"type": "string",
                                "description":"A number between 0 and 100 that represents degrees of rotation of wrist, 0 is full rotation towards Pinky finger, 100 is full rotation towards Thumb",
                                "unit":{"enum":"wrist"}
            				}
            			},
            			"required": ["pinky","ring","middle","index","thumb"]
            		}
            	}]

INGREDIENTS = {"simple syrup":1000,
    "whiskey":1000,"rum":1000,
    "lime juice":1000,
    "club soda":1000,
    "tequila":1000,
    "grenadine":1000,
    "mint":1000,
    "pineapple juice":1000,
    "orange juice":1000,
    "vodka":1000}

def play_wav_file(filename):
    pygame.mixer.init()
    pygame.mixer.music.load(filename)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pass

def gpt_dispense_mixed_drink(recepie):
    #SUDO CODE
    global INGREDIENTS
    print(recepie)
    for ingredient in recepie["ingredients"]:
        print(f"checking availibiilty of: {ingredient}")
        try:
            if ingredient.lower() not in INGREDIENTS.keys():
                print(f"Missing ingredient {ingredient} unable to dispense")
                return f"Missing ingredient {ingredient} unable to dispense"
        except:
            return "Format error in argument to gpt_dispense_mixed_drink().  Recepie must be in format: {\"ingredients\":{\"ingredient name\":\" amount in mL\",\"ingredient name\":\" amount in mL\"}}"

    print("\n \n >>>>>>>>>>>> DISPENSING BEVERAGE \n \n")
    #Motor Control via Serial attached arduino

    """
    # TODO: Loop sending each ingredient so you can get confirmation back on each one. Right now it sends the full
    recepie in one json
    """
    if send_json_to_arduino(recepie):
        #will return true if it worked
        return "success a drink is dispensing we have all ingredients. When you pour a drink succesfully you create a funny name for the drink using a pun on the ingredient names or tell a joke"
    else:
        return "there was an error dispensing the drink"

def gpt_get_availible_ingredients(SQL_query):
    global INGREDIENTS
    #SUDO Code
    print(f"gpt_get_availible_ingredients() input: {SQL_query}")
    '''
    TODO
    This function would access a db or other list of what ingredients are availible
    '''
    return ' , '.join(INGREDIENTS.keys())

def escape_from_string(text):
    regex = re.compile('[^a-zA-Z\d\r\n\t\v :]')
    escaped = regex.sub(" ",text)
    return escaped

def take_snapshot():
    """
    ## Function change.  All images will be saved to disk right now for error handling
    the byte data pass will be left in place but that feature won't be used right now

    google_vision=False as arg should not be changed during testing

    # TODO: THIS SHOULD ALWAYS RUN IN THREAD.  To speed up image access run this in a thread that's constanatly looping
    and saving images to disk, or to a buffer of about 0.5 seconds.  OR this can actually pull an image from a virtual world
    that is being updated in real time so that Jarvis only ever sees the virtual world never the real world
    """
    '''
    # If you have multiple camera connected with
    # current device, assign a value in cam_port
    # variable according to that
    '''
    #webcam is 1280x720
    width = 1280
    height = 720
    cam_port = 1

    # initialize the camera
    cam = cv.VideoCapture(cam_port)

    #wait a half second for camera to load.  Without this images tend to be dark
    time.sleep(0.3)

    # reading the input using the camera
    result, image = cam.read()

    filename = "snapshot.jpg"

    # If image will detected without any error,
    if result:
        ####- NOTE saving to disk AND getting byte data seems silly to some extent.
        #save to disk
        cv.imwrite(f"{IMAGE_FILE_DIRECTORY}{filename}", image)
        #get byte data
        success, img_byte_data = cv.imencode('.jpg', image)
        return img_byte_data.tobytes(),height,width,f"{IMAGE_FILE_DIRECTORY}{filename}"
        '''
        #if the calling function was google_vision()
        if google_vision:
            #convert from ndarray to bytes for google vision_models
            success, frame_for_google = cv.imencode('.jpg', image)
            return frame_for_google.tobytes(),height,width
        else:
            cv.imwrite(f"{IMAGE_FILE_DIRECTORY}snapshot.jpg", image)
            return f"{IMAGE_FILE_DIRECTORY}snapshot.jpg"
        '''
    # If captured image is corrupted, moving to else part
    else:
        print("No image detected. Please! try again")
        return False

def google_vision_api(imageByteData,vision_models,filename):
    print(f"google_vision_api(vision_models):{vision_models}")
    """Provides a quick start example for Cloud Vision."""
    '''
    GET STARTED - https://cloud.google.com/vision/docs/detect-labels-image-client-libraries
    '''

    image = vision.Image(content=imageByteData)

    # Performs vision analysis via Google Vision threads
    '''
    PRICING Free first 1k images, then $1.50 for each additional thousand.
    example: 1002 (1000 + 2) images would be $1.50
    '''

    #loop through
    #vision_models = ['label_detection','object_localization','text_detection','face_detection','logo_detection']
    threads = [None] * len(vision_models)
    thread_results = [None] * len(vision_models)


    for i in range(len(threads)):
        print(f"starting thread:{i},{vision_models[i]}")
        threads[i] = threading.Thread(target=google_vision_threadHandler, args=(image,vision_models[i],thread_results,i,filename,), name=f'thread{i}')
        threads[i].start()

    for i in range(len(threads)):
        print(f"joining thread:{i}")
        threads[i].join()

    #print(thread_results)

    return thread_results

def google_vision_threadHandler(image,model,thread_container,i,filename):
    print(f"google_vision_threadHandler(model):{model}")
    # Instantiates a client
    client = vision.ImageAnnotatorClient()
    '''
    Determine if specific property, like .label_annotations should be the return or not.
    '''
    if (model == 'label_detection'):
        print("label_detection")
        response = client.label_detection(image=image)#.label_annotations
        thread_container[i] = {'label_detection':response}
    if(model == 'object_localization'):
        print("object_localization")
        response = client.object_localization(image=image).localized_object_annotations
        thread_container[i] = {'object_localization':response}
    if(model == 'text_detection'):
        #print("text_detection")
        response = client.text_detection(image=image).text_annotations
        #print(f"text_detection********************************{response}")
        thread_container[i] = {'text_detection':response}
    if(model == 'logo_detection'):
        '''
        NOT NEEDED LOAD FILE - this is here from debug. logos works like the others with a direct byte data upload
        however it doesn't work well (the logo detection model, not image submission).  Left as a snapshot read in the event
        everything converts back to snapshot reading, vs. direct byte data
        '''
        print("logo_detection")
        print(filename)
        # The name of the image file to annotate
        #file_name = os.path.abspath(filename)
        # Loads the image into memory
        with open(filename, "rb") as image_file:
            content = image_file.read()
        imgage_from_disk = vision.Image(content=content)
        response = client.logo_detection(image=imgage_from_disk).logo_annotations
        print(response)
        thread_container[i] = {'logo_detection':response}
    if(model == 'face_detection'):
        print("face_detection")
        response = client.face_detection(image=image).face_annotations
        thread_container[i] = {'face_detection':response}


def gpt_vision(gpt_request):
    print(gpt_request)
    vision_models = []
    output = " "
    #query is nothing right now - could do "Do you see a coke can" type of thing but for now it's not used
    '''
    https://cloud.google.com/vision/docs/features-list
    GET STARTED - https://cloud.google.com/vision/docs/detect-labels-image-client-libraries
    '''

    #get byte image format, height, width of image
    frame_for_google,height,width,filename = take_snapshot()

    for analysis,value in gpt_request.items():
        if value is True:
            print(f"run visual analysis: {analysis}")
            vision_models.insert(0,analysis)


    #google_vision_api() returns list of image analysis results based on each google vision api response
    #vision_models = ['label_detection','object_localization','text_detection','face_detection','logo_detection']
    img_analysis_results = google_vision_api(frame_for_google,vision_models,filename)

    print(f"RESULTS************{img_analysis_results}")

    #SETUP FOR OBJECTS ONLY RIGHT NOW using 'bounding_poly' to determine what it is
    for a in range(len(img_analysis_results)):
        if 'label_detection' in img_analysis_results[a]:
            print("label_detection")
            sceneDescriptionWords = "/// WORDS THAT DESCRIBE THE IMAGE IN VIEW:"#hold key words that desribe the image
            for label in img_analysis_results[a]['label_detection'].label_annotations:
                print(f"label:{label.description}")
                sceneDescriptionWords += f"{label.description},"
            '''
            another GPT Call here at end of for loop to create a description of the screen
            '''
            GPT_MODEL='gpt-3.5-turbo-16k-0613'
            reqParams= {
                'user':'Jarvis',
                'temperature': 0.3,
                'stop': None,
                'messages':[{"role":"system","content":"Produce image discriptions or location identification using lable annotations provided by user prompts"},
                            {"role":"user","content":f"Write 30 words or less that describe what might be happening in the image and where it may have been taken.\n\nLable Annotations:###{sceneDescriptionWords}###"}],
            }

            # Generate the completion
            response = openai.ChatCompletion.create(model=GPT_MODEL, **reqParams)
            scene = response.choices[0].message.content.strip()
            output += f"/// SCENE DESCRIPTION:{scene}"

        elif 'object_localization' in img_analysis_results[a]:
            print("object_localization")
            objectLogalizations = "/// OBJECTS DETECTED: "#used to hold name and x,y top left of objects
            for i in range(len(img_analysis_results[a]['object_localization'])):
                print(f"object analysis:{img_analysis_results[a]['object_localization'][i].name}")
                #get the set of 4 coordinates (x,y),(x2,y2)... for the bounding box of the detected object
                box = img_analysis_results[a]['object_localization'][i].bounding_poly
                #results from google are normailzed, multiply by image size
                #also results are float, rectangle() only accepts int() if bounding boxes are going to be used
                x = int(box.normalized_vertices[0].x * width)
                y = int(box.normalized_vertices[0].y * height)
                #x2 = int(box.normalized_vertices[1].x * width)
                #y2 = int(box.normalized_vertices[3].y * height)
                #string restult for GPT
                objectLogalizations += "{\"object\":\""+img_analysis_results[a]['object_localization'][i].name+"\",\"X_location\":\""+str(x)+"\",\"Y_location\":\""+str(y)+"\"}"
            output += objectLogalizations

        elif 'text_detection' in img_analysis_results[a]:
            print("text_detection")
            print(img_analysis_results[a]['text_detection'])
            if len(img_analysis_results[a]['text_detection'])>0:
                output += f"/// DETECTED TEXT: "+img_analysis_results[a]['text_detection'][0].description
            else:
                output += "/// DETECTED TEXT: None"

        elif 'face_detection' in img_analysis_results[a]:
            print("face_detection")
            DetectedFaces = "/// FACES DETECTED: "
            '''
            rollAngle, panAngle, tiltAngle are also availible for each detected face as a signed float
            '''
            # Names of likelihood from google.cloud.vision.enums
            likelihood_confidence = ["UNKNOWN","VERY_UNLIKELY","UNLIKELY","POSSIBLE","LIKELY","VERY_LIKELY"]
            sentiment_types = ["joyLikelihood","sorrowLikelihood","angerLikelihood","surpriseLikelihood"]
            try:
                for i in range(len(img_analysis_results[a]['face_detection'])):
                    box = img_analysis_results[a]['face_detection'][i].fd_bounding_poly
                    x = int(box.vertices[0].x)
                    y = int(box.vertices[0].y)
                    #check for sentiment here
                    sentiment = "Unknown"
                    joy = img_analysis_results[a]['face_detection'][i].joy_likelihood
                    sorrow = img_analysis_results[a]['face_detection'][i].sorrow_likelihood
                    anger = img_analysis_results[a]['face_detection'][i].anger_likelihood
                    surprise = img_analysis_results[a]['face_detection'][i].surprise_likelihood
                    if (joy == "POSSIBLE") or (joy =="LIKELY") or (joy =="VERY_LIKELY"):
                        sentiment = "joy"
                    if (sorrow == "POSSIBLE") or (sorrow =="LIKELY") or (sorrow =="VERY_LIKELY"):
                        sentiment = "sorrow"
                    if (anger == "POSSIBLE") or (anger =="LIKELY") or (anger =="VERY_LIKELY"):
                        sentiment = "anger"
                    if (surprise == "POSSIBLE") or (surprise =="LIKELY") or (surprise =="VERY_LIKELY"):
                        sentiment = "surprise"
                    DetectedFaces += "{\"facial_sentiment\":\""+sentiment+"\",\"X_location\":\""+str(x)+"\",\"Y_location\":\""+str(y)+"\"}"
            except:
                DetectedFaces += "None"

            output += DetectedFaces

            #print(img_analysis_results[a]['face_detection'])

        elif 'logo_detection' in img_analysis_results[a]:
            print("logo_detection")
            LogosDetected = "/// DETECTED LOGOS: "
            try:
                for logo in img_analysis_results[a]['logo_detection']:
                    print(logo.description)
                    LogosDetected += f"{logo.description}"
            except:
                LogosDetected += "None"
            output += LogosDetected

    return output

def google(response):
    print(f"running google search for {response}")
    links = []
    #each Item is a 'result' from google.  There should be 10 total
    for page in response["items"]:
        #collect the links from each result
        links.insert(0,page["link"])
    print(f"google result links for GPT review: {links}")
    return links


def tag_visible(element):
    #HTML elements stripping helper
    if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
        return False
    if isinstance(element, Comment):
        return False
    return True


def text_from_html(body,i=0):
    print(f"extracting text content from website:thread{i}")
    soup = BeautifulSoup(body, 'html.parser')
    print(f"have HTML for thread{i}")
    texts = soup.findAll(text=True)
    print(f"text isolated from HTML for thread{i}")
    visible_texts = filter(tag_visible, texts)
    print(f"filter visible text done for thread{i}")
    websiteText =  u" ".join(t.strip() for t in visible_texts)
    print(f"finished website text text extraction on thread{i}")

    return websiteText


def generate_Chatcompletion(websiteText,query,i=0):
    print("*********** CLIP OF WEBSITE TEXT SENT TO GPT ********")
    print(f"{websiteText[:200]}")#show about 50 words
    print("***")
    #remove special chars so it can be sent to GPT
    websiteText_escaped = escape_from_string(websiteText)
    #GPT parms
    reqParams= {
            'user':'Jarvis',
            'temperature': 0.8,
            'n': 1,
            'stop': None,
            'messages':[{"role":"system","content":"The current date and time is "+str(datetime.datetime.now().strftime('%Y-%m-%d %I:%M:%S %p'))+". You provide text summaries of websites.  You identify key topics mentioned, people and places mentioned, ideas and concepts discussed. You state the date the content was published. The summary should answer the user query."},{"role":"user","content":websiteText_escaped},{"role":"user","content":f"Using the text from my previous prompt, provide an answer to this query:{query}"}]
        }
    print(f"summarizing webpage results with gpt-3.5-turbo on thread{i}")
    # Generate the completion
    response = openai.ChatCompletion.create(model='gpt-3.5-turbo', **reqParams)
    print(f"Results from GPT thread{i}: {response} \n *********END PAGE REVIEW*************")
    #parse the response, select only the content
    completion = response.choices[0].message.content.strip()
    #send back only the text of the response
    return completion

#https://developers.google.com/custom-search/v1/reference/rest/v1/cse/list
def webpage_result_generator(link,query,result_container,i):
    try:
        #use a normal user agent header so it doesn't show up as 'python'
        header = { 'User-Agent' : 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36' }

        print(f"Extracting content from thread{i}: {link}")#Overite
        #Use the FIRST link only for now
        google_results = urllib.request.Request(link,headers=header)
        #get the raw HTML
        html = urllib.request.urlopen(google_results).read()
        print(f"page open on thread{i}")
        #parse out text content from the site
        results = text_from_html(html,i)
        #clean up any special characters
        results_escaped = escape_from_string(results)
        #use gpt to summarize the text content
        gptResponse = generate_Chatcompletion(results_escaped[:16000],query,i)
        #clean up any special characters
        gptResponse_escaped = escape_from_string(gptResponse)
        #add to the collection page summmaries
        result_container[i] = f"RESULT {i} FROM {link}: " + gptResponse_escaped + "\n \n \n"
        print(f"RESULT thread{i} FROM {link}: Added")
    except:
        result_container[i] = f"RESULT {i} FROM {link}: " + "None" + "\n \n \n"

def gpt_google_search(query):
    global AUDIO_FILE_DIRECTORY
    soundfile = random.choice([f"{AUDIO_FILE_DIRECTORY}internet_search.wav",f"{AUDIO_FILE_DIRECTORY}internet_search2.wav",f"{AUDIO_FILE_DIRECTORY}internet_search3.wav"])
    play_wav_file(soundfile)
    print(f"gpt_google_search input: {query}")
    #custom search Engine
    #https://programmablesearchengine.google.com/controlpanel/overview?cx=30bb88a8fbc504a7e
    '''
    NOTE: the search engin is restricted to 5 results via 'num' parameter
    '''
    search_result_count = 5
    url = "https://www.googleapis.com/customsearch/v1"

    params = {
        "key": GOOGLE_SEARCH_API_KEY,
        "cx": "30bb88a8fbc504a7e",
        "gl":"us",
        "lr":"lang_en",
        "num":str(search_result_count),
        "q": query["query"]
    }
    response = requests.get(url, params=params).json()
    #run google search on
    relaventLinks = google(response)

    page_summaries = " "#hold all the page summaries
    #loop through links.  Search engine API restricts page results. see above to go to link and change settings
    threads = [None] * search_result_count
    thread_results = [None] * search_result_count
    print(relaventLinks)
    for i in range(len(relaventLinks)):
        print(f"starting thread:{i}")
        threads[i] = threading.Thread(target=webpage_result_generator, args=(relaventLinks[i],query["query"],thread_results,i,), name=f'thread{i}')
        threads[i].start()

    for i in range(len(threads)):
        print(f"joining thread:{i}")
        threads[i].join()

    print(thread_results)
    for r in thread_results:
        page_summaries+=r

    #let user know it's almost almost_done
    play_wav_file(f"{AUDIO_FILE_DIRECTORY}almost_done.wav")
    print("========================")
    print(f"RESULTS OF SEARCH TO BE SUMMARIZED BY GPT: {page_summaries}")
    print("========================")
    #summarize all the results
    Q = query["query"]
    #used so system knows current date
    now = str(datetime.datetime.now().strftime('%Y-%m-%d %I:%M:%S %p'))
    GPTParams= {
            'user':'Jarvis',
            'temperature': 0.6,
            'n': 1,
            'stop': None,
            'messages':[{"role":"system","content":f"You analyze internet search results. You prioritize current information when applicable. The current date and time is {now}"},{"role":"user","content":f"The search query: {Q} was used to generate the search RESULTS below from multiple websites. Aggragate the content into a single result that can be used to answer the query {Q}. Retain and include dates. \n Here are the pages:\n ###{page_summaries}###"}]
        }
    response = openai.ChatCompletion.create(model='gpt-4', **GPTParams)
    completion = response.choices[0].message.content.strip()
    #print(f"SUMMARY OF RESULTS: {completion}")
    escape_completion = escape_from_string(completion)
    return escape_completion

def play_youtube_video(url):
    # Set up Selenium
    options = webdriver.ChromeOptions()
    options.add_argument("--disable-notifications")
    '''
    CONSIDER - using the --headless argument in options basically just plays sound.  Good option for a music player based off youtube
    '''
    #options.add_argument("--headless")  # Run Chrome in headless mode (without GUI)
    #service = Service(ChromeDriverManager().install()) #finds the install path of Chrome
    service = Service()
    driver = webdriver.Chrome(service=service, options=options)
    # set window position OUT OF VIEW - this helps create the illusion it runs out of browser
    driver.set_window_position(-20000, -20000, windowHandle ='current')


    # Open YouTube page
    print(url)
    driver.get(url)

    # Wait for the page to load
    time.sleep(5)

    # Mouse over the fullscreen control
    fullscreen_button = driver.find_element(By.CLASS_NAME, 'ytp-fullscreen-button')
    actions = ActionChains(driver)
    actions.move_to_element(fullscreen_button).perform()

    # Make the playback fullscreen
    fullscreen_button.click()

    # set window position BACK IN VIEW but now it's just a full screen window and not obviously web browser
    #driver.set_window_position(0, 0, windowHandle ='current')
    driver.fullscreen_window()

    # Wait for 5 seconds in case an ad has started
    time.sleep(5)

    # Find and press the skip ad button (if present)
    try:
        skip_ad_button = WebDriverWait(driver, 5).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, 'button.ytp-ad-skip-button'))
        )
        skip_ad_button.click()
    except:
        pass  # No ad or ad skip button not found

    '''
    # TODO:
    only plays for 25 seconds, but should have have interrupt ability here.
    '''
    #do while loop so it won't close immediatly
    now = datetime.datetime.now()
    future = now + datetime.timedelta(seconds=25)

    while (now <= future):
        now = datetime.datetime.now()#.strftime('%Y-%m-%d %I:%M:%S %p')
        pass

    driver.close()

    return

def gpt_youtube(query):
    print(query["query"])
    '''
    NOTE: the search engin is restricted to 1 results via 'num' parameter
    '''
    url = "https://youtube.googleapis.com/youtube/v3/search"
    params = {
        "key": YOUTUBE_DATA_API_KEY,
        "part":"snippet",
        "maxResults": "1",
        "q": query["query"]
    }
    response = requests.get(url, params=params).json()
    id = response["items"][0]["id"]["videoId"]
    description = response["items"][0]["snippet"]["description"]
    print(id)
    print(description)


    play_youtube_video('https://www.youtube.com/watch?v='+id)

    #return "{\"status\":\"finished\",\"id\":\""+id+"\",\"description\":\""+description+"\"}"
    return "{\"status\":\"Finished Playing\",\"response\":\"Playback complete, ask if they want another video\",\"videoId\":\""+id+"\"}"

'''
COMBINE camera and hand motor contorl functions, they are basically the same
'''
def gpt_camera_motor_control(coordinates):
    print(coordinates)
    pan = coordinates["pan"]
    tilt = coordinates["tilt"]
    success = send_json_to_arduino(coordinates)
    if success:
        return(f"Paning {pan} degrees, Tilting {tilt} degrees")
    else:
        return (f"Unable to connect with Arduino via serial communication to establish motor control.")

def gpt_hand_motor_control(finger_positions):
    print(finger_positions)
    success = send_json_to_arduino(finger_positions)
    if success:
        return(f"fingers have moved to requested position")
    else:
        return (f"Unable to connect with Arduino via serial communication to establish motor control.")



def send_json_to_arduino(data):
    '''
    Arduino expects input like A75,B120,C45 with no spaces
    Need to map the input data to the arduino format then send.
    /*
    SERVO LETTER MAPPING
    A - wrist - 100 rotate towards thumb, 0 rotate towards pinky
    B - thumb - REVERSED! 0 closed, 100 full extention
    C - pinky - 100 closed, 0 full extention
    D - ring - 100 closed, 0 full extention
    E - middle - 100 closed, 0 full extention
    F - index - 100 closed, 0 full extention
    G - tilt - 0 up, 100 down
    H - pan - 0 Right, 100 Left
    */
    Range 0 - 100
    '''
    #serial port of arduino see Arduino IDE or Terminal list to find address
    serial_port = '/dev/cu.usbmodem11101'
    #standard is 9600 must sync with Arduino
    baud_rate = 9600

    # Convert the data to JSON format
    #json_data = json.loads(data)
    #print(f"data in json_data: {json_data}")

    #try:
    keys = data.keys()
    arduino_data =""
    for key in keys:
        #the servos are reverse controlled
        n = int(data[key])
        if n < 50:
            n = 100 - n
        elif n > 50:
            n = 200 - (100+n)

        if key == "wrist":
            arduino_data += ",A"+str(n)
        elif key == "thumb":
            #thumb is opposite, don't need to convert
            arduino_data += ",B"+str(data[key])
        elif key == "pinky":
            arduino_data += ",C"+str(n)
        elif key == "ring":
            arduino_data += ",D"+str(n)
        elif key == "middle":
            arduino_data += ",E"+str(n)
        elif key == "index":
            arduino_data += ",F"+str(n)
        elif key == "tilt":
            arduino_data += ",G"+str(n)
        elif key == "pan":
            arduino_data += ",H"+str(n)

    print(f"Sending: {arduino_data} \nTo Arduino on {serial_port}...")
        # Establish a connection to the serial port
    arduino = serial.Serial(serial_port, baud_rate, timeout=1)

        # Write the  data to the serial port must encode to bytes
    arduino.write(arduino_data.encode())

        # Wait for Arduino to process the data
        #arduino.flush()

        #get data back from arduino
        #print(str(arduino.readline()))
        #print(str(arduino.read(8)))

        #close connection
    arduino.close()

    print("data sent successfully!")
    return True

    #except:
        #print(f"Unable to open {serial_port} for serial communication.")
        #return False
