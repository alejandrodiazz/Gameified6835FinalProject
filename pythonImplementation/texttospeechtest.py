# Import the required module for text 
# to speech conversion
from gtts import gTTS
  
# This module is imported so that we can 
# play the converted audio
import os
  
def convert_dict_to_files(prefix, text_dict):
	
  	for i in text_dict:
  		filename = prefix+i
  		mytext = text_dict[i]
  		language = 'en'

		# Passing the text and language to the engine, 
		# here we have marked slow=False. Which tells 
		# the module that the converted audio should 
		# have a high speed
  		myobj = gTTS(text=mytext, lang=language, slow=False)

  		myobj.save(filename)
		# Playing the converted file
	  	os.system("mpg321 "+ filename)

# The text that you want to convert to audio
text_to_convert = {"beep.mp3":"go", "welcome.mp3": "Welcome to Gameified!", "fail.mp3": "Fail.", "goodjob.mp3":"Good Job!", "goodrep.mp3":"Perfect", "couldbebetter.mp3":"Good", "decentrep.mp3": "Okay", "terriblerep.mp3": "Not so good"}

convert_dict_to_files("audio/", text_to_convert)