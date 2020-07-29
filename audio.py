from pydub import AudioSegment
from pydub.playback import play
from pydub import AudioSegment
from pydub.playback import play
import sys
import os
from pydub.utils import make_chunks
import numpy as np
from scipy.fftpack import fft
import pandas as pd
import matplotlib.pyplot as plt
import glob
import seaborn as sns
# import machile learning 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import math
from sklearn.metrics import mean_squared_error
from sklearn.experimental import enable_hist_gradient_boosting  
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.experimental import enable_hist_gradient_boosting  
from sklearn.datasets import load_diabetes
from sklearn.neighbors import KNeighborsClassifier
from sklearn  import tree
from sklearn.model_selection import GridSearchCV
import pyaudio
import wave
from scipy.io import wavfile
from sklearn import preprocessing
from src.fourier import audioFeaturesFourie

# import messages 
import os 
from twilio.rest import Client
import dotenv
dotenv.load_dotenv()
#from audio import reconocedor_audio

# FOURIER 

#def audioFeaturesFourie(audio):
 #   array = audio.get_array_of_samples()
  #  abs_four = np.abs(fft(array,n=512))
   # return abs_four

# AUDIOS 
print(' Cargando Primer audio')
cafe = AudioSegment.from_file("./audio/Cafe_try.mp3")
cafe_change_hz = cafe.set_frame_rate(16000)
chunks=make_chunks(cafe_change_hz, 5000)
cafe_tf = [audioFeaturesFourie(x) for x in chunks]
print ("Audio de Cafe Cargado")

print(' Cargando Segundo audio')
lavadora = AudioSegment.from_mp3("./audio/Lavadora_trybueno.mp3")
lavadora_change_hz = lavadora.set_frame_rate(16000)
chunks=make_chunks(lavadora_change_hz, 5000)
lavadora_tf = [audioFeaturesFourie(x) for x in chunks]
print ("Audio de Lavadora Cargado")

print(' Cargando Tercer  audio')
aspiradora = AudioSegment.from_mp3("./audio/aspiradora _try.mp3")
aspiradora_change_hz = aspiradora.set_frame_rate(16000)
chunks=make_chunks(aspiradora_change_hz, 5000)
aspiradora_tf = [audioFeaturesFourie(x) for x in chunks]
print ("Audio de Aspiradora Cargado")
"""
print(' Cargando Cuarto  audio')
silencio = AudioSegment.from_mp3("./audio/Silencio_try_bueno.mp3")
silencio_change_hz = silencio.set_frame_rate(16000)
chunks=make_chunks(silencio_change_hz, 5000)
silencio_tf = [audioFeaturesFourie(x) for x in chunks]
print ("Audio de Silencio Cargado")
"""
print(' Cargando Cuarto  audio')
thermomix = AudioSegment.from_mp3("./audio/Thermomix_try.mp3")
thermomix_change_hz = thermomix.set_frame_rate(16000)
chunks=make_chunks(thermomix_change_hz, 5000)
thermomix_tf = [audioFeaturesFourie(x) for x in chunks]
print ("Audio de Thermomix Cargado")
"""
print(' Cargando quinto  audio')
noise = AudioSegment.from_file("./audio/noise.webm")
noise_change_hz = cafe.set_frame_rate(16000)
chunks=make_chunks(cafe_change_hz, 5000)
noise_tf = [audioFeaturesFourie(x) for x in chunks]
print ("Audio de Ruido Cargado")
"""

# DATAFRAME 
print ("Creando DataFrame")
Cafe = {"Sound": [a for a in cafe_tf], "utensilio": 1}
Lavadora = {"Sound": [a for a in lavadora_tf], "utensilio": 2}
Aspiradora = {"Sound": [a for a in aspiradora_tf], "utensilio": 3}
#Silencio = {"Sound": [a for a in silencio_tf], "utensilio": "silencio"}
Thermomix = {"Sound": [a for a in thermomix_tf], "utensilio": 4}
#Noise = {"Sound": [a for a in noise_tf], "utensilio": 5}

cafe = pd.DataFrame(Cafe)
lavadora= pd.DataFrame(Lavadora)
aspiradora= pd.DataFrame(Aspiradora)
#silencio = pd.DataFrame(Silencio)
thermomix = pd.DataFrame(Thermomix)
#noise = pd.DataFrame(Noise)

utensilio_df = pd.concat([cafe,lavadora,aspiradora,thermomix]).reset_index(drop=True)
print ("Dividiendo DataFrame en X e y")
X =np.concatenate((np.vstack(cafe.Sound),np.vstack(lavadora.Sound),np.vstack(aspiradora.Sound),np.vstack(thermomix.Sound)))
y = np.concatenate((cafe.utensilio,lavadora.utensilio,aspiradora.utensilio,thermomix.utensilio))

print ("Empezando Bucle")

while True:
    result={1:'cafetera',2:'lavadora',3:'aspiradora',4:'thermomix', 5:"noise"}
    CHUNK = 5000
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    RECORD_SECONDS = 5
    WAVE_OUTPUT_FILENAME = ("record/prueba_silencio.wav")
    y_pref_sol=[]
    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

    print("* recording")

    frames = []

    for i in range(0, int(16000 / 5000 * 5)):
        data = stream.read(5000)
        frames.append(data)


    print("* done recording")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    print ("Pasando el audio al modelo para predecir electrodomestico")
    cafe_prueba = AudioSegment.from_file("./record/prueba_silencio.wav")
    cafe_change_hz = cafe_prueba.set_frame_rate(16000)
    chunks=make_chunks(cafe_change_hz, 5000)
    cafe_tf = [audioFeaturesFourie(x) for x in chunks]
    cafe_prueba2=np.array(cafe_tf)
    cafe_prueba2.shape
    npcafe = np.array(cafe_tf)   
    print(cafe_prueba)
    model =  HistGradientBoostingClassifier()
    model.fit(X,y)
    y_pred=model.predict(cafe_prueba2)
    y_pref_sol.append(y_pred)

    client = Client(os.getenv("Account_Sid"),
    os.getenv("Auto_Token"))


    message = client.messages.create(
    to = os.getenv("Cell_phone"),
    from_ = os.getenv("Phone_number"),
    body = f"El electrodomestico identificado es : {result[int(y_pref_sol[-1])]}")

    print(message.sid)
    
        #print(f" El electrodomestico identificado es : {result[int(y_pref_sol[-1])]}")
    orden=input('para o seguir: ')
        
    if orden =='para':
        break
        