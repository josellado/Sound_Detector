import os
from flask import Flask, flash, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename

#app.run("0.0.0.0", port = 5000, debug =True)

from flask import Flask, render_template, request
import subprocess

app = Flask(__name__)

@app.route('/')
def init_recorder():
    return render_template('VoiceRecorder.html')
    
@app.route('/uploads', methods=['POST'])
def save_audio():
    rawAudio = request.get_data()
    audioFile = open('RecordedFile.wav', 'wb')
    audioFile.write(rawAudio)
    audioFile.close()
    return speech_to_text()
    
def speech_to_text():
    subprocess.run('python3 audio.py', shell=True)
    inFile = open("/record/prueba_silencio" ,'r')
    transcript = ''
    for line in inFile:
        transcript += line
    print(transcript)
    return transcript
    
if __name__ == '__main__':
    app.run("0.0.0.0" , debug=True, port=5000)



