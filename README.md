# Gameified6835FinalProject
This is our final project for 6.835 Multimodal Interfaces and Programming at MIT.\

We will use a Kinect to track and give will time feedback as people complete exercises such as push-ups and squats.

All of the files for running the python is in pythonimplementation. Everything else was just code we tried at the beginning.  

Important Folders/Files (all others can be ignored):

Folder: pythonImplementation
  Folder/Files within pythonImplementation: 
    getTrainerData.py - this file is used to process training data. If the trainer wants to add a new exercise, they should film a video and then process the video using this file to get csv training data. 
    implementation.py - this is the main python file with all the main game logic/ UI / accuracy calculator. Run this to use Gameified.
    videos - this is a folder containing all trainer videos
    audio - this folder contains all audio recordings used during the game
    .csv - all csv files are training data collected from running getTrainerData.py on videos
    videooverlay.py - this file supports playing the trainer video on top of the Gameified UI 
    poseModule.py - 3D skeletal code used to detect pose and positions in order to overlay a skeleton and perform accuracy calculations
    texttospeech.py - this code converts strings to audio mp3 files
    implementation_threaded.py - can be ignored, for debugging purposes
  
Notes:

I had to run "brew install mpg321" to download the command for the audio text to speech to work.

It is likely you may have to install several libraries/packages via pip install or some equivalent. But you should be prompted to do so. 

Also had to run these commands to install

pip install playsound

sudo pip install pyobjc
