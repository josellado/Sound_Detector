import os
from flask import Flask, flash, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
"""
app = Flask(__name__) 

UPLOAD_FOLDER = "./record/prueba_silencio"
ALLOWED_EXTENSIONS = {"wav"}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('uploaded_file',
                                    filename=filename))
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    


"""

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
    subprocess.run('python3 speechtotext.py', shell=True)
    inFile = open("/record/prueba_silencio" + 'result/result.txt', 'r')
    transcript = ''
    for line in inFile:
        transcript += line
    print(transcript)
    return transcript
    
if __name__ == '__main__':
    app.run("0.0.0.0" , debug=True, port=5000)



