from openai import OpenAI
import requests
import base64
import json
import numpy as np
from geopy.geocoders import Nominatim
import sounddevice as sd
import cv2
import time
import os
import csv
import pandas as pd
import joblib

from colorama import init, Fore, Back, Style
from tabulate import tabulate
from metpy.calc import dewpoint_from_relative_humidity, dewpoint
from metpy.units import units
from ultralytics import YOLO
from multiprocessing import Process
from datetime import datetime, timedelta
from mediapipe.tasks import python
from mediapipe.tasks.python.components import containers
from mediapipe.tasks.python import audio
from scipy.io import wavfile
from scipy.io import wavfile
from scipy.io.wavfile import write as wavwrite

init(autoreset=True)

CameraURL = ''
ApiUrl = 'http://10.19.71.68:1337'
ApiKey = ''
PromptGPTCamera = '''
Eres un sistema de análisis de imágenes especializado en detección temprana de incendios forestales. 
Analiza la imagen proporcionada y determina:

- Si hay presencia de humo.
- Si hay presencia de fuego.
- Nivel de certeza de cada detección (en %).
- Breve descripción visual de la escena.

Devuelve el resultado estrictamente en formato JSON con la siguiente estructura:

{
  "humo": booleano,
  "fuego": booleano,
  "certeza_humo_porcentaje": número,
  "certeza_fuego_porcentaje": número,
  "descripcion_visual": "texto breve"
}

Reglas:
- Si no se detecta humo o fuego, retorna false y un porcentaje de certeza bajo.
- No incluyas explicaciones fuera del JSON.
- El JSON debe ser válido.

'''
PromptGPTReport = '''
Eres un sistema experto en detección temprana de incendios forestales. 
Tu tarea es analizar datos de sensores ambientales y entregar un informe en formato JSON.

Datos que recibirás:
- MQ-2 (PPM de gases combustibles y humo)
- Velocidad del viento (km/h)
- Temperatura (°C)
- Humedad (%)
- Latitud (lat)
- Longitud (lon)
- Dirección del viento (grados o cardinal)

Requisitos del análisis:
1. Evalúa la probabilidad de incendio forestal en base a los datos.
2. Identifica condiciones críticas (alta temperatura, baja humedad, alta concentración de humo/gases, viento fuerte).
3. Genera un nivel de alerta: "Bajo", "Medio", "Alto", "Crítico".
4. Incluye recomendaciones inmediatas (ejemplo: monitoreo, alerta preventiva, despacho de brigada).
5. Usa siempre JSON bien estructurado, sin texto adicional fuera del JSON.
6. Solo di lo necesario.

Formato de salida JSON:
{
  "ubicacion": {
    "lat": <valor>,
    "lon": <valor>
  },
  "condiciones": {
    "temperatura_c": <valor>,
    "humedad_%": <valor>,
    "mq2_ppm": <valor>,
    "velocidad_viento_ms": <valor>,
    "direccion_viento": "<valor>"
  },
  "analisis": {
    "riesgo_incendio": "<Bajo|Medio|Alto|Crítico>",
    "factores_determinantes": ["<lista de factores>"],
    "recomendaciones": ["<acciones sugeridas>"]
  },
  "timestamp": "<fecha-hora UTC>"
}
'''
banner = '''
_________                __  .__              .__          
\\_   ___ \\  ____   _____/  |_|__| ____   ____ |  | _____   
/    \\  \\/_/ __ \\ /    \\   __\\  |/    \\_/ __ \\|  | \\__  \\  v1.0-alpha
\\     \\___\\  ___/|   |  \\  | |  |   |  \\  ___/|  |__/ __ \\_
 \\______  /\\___  >___|  /__| |__|___|  /\\___  >____(____  /
        \\/     \\/     \\/             \\/     \\/          \\/ 
 🌲🔥 Deteccion temprana de Incendios Forestales usando Machine Learning,
 Vision Artificial, y sensores de temperatura y humedad.

 Desarrollado por:                        Junto a:
    - Tomás Benitez  - Agustin Troncoso      - Profesora Daniela Reyes
    - Alonso Ramirez - Tomás Gacitúa         - Estimado Cristobal Siñiga

'''

client = OpenAI(api_key=ApiKey)
model = YOLO('models/FireAndSmoke.pt')


def rule_thirty(temperature: int, humidity: int, wind_speed: int):
    if temperature >= 30 and humidity <= 30 and wind_speed >= 30:
        return 0
    
    elif temperature >= 25 and humidity <= 35 and wind_speed >= 25:
        return 1
    
    else:
        return 2


def get_sensor_data():
    response = requests.get(ApiUrl + '/api/sensor_data').json()[0]

    return response

def get_address(lat, lon):
    geolocator = Nominatim(user_agent="Centinela/1.0.0")
    location = geolocator.reverse((lat, lon), language='es')
    if location:
        return location.address
    else:
        return None
        
def chainsaw_detect(audioFile: str):
    possible_chainsaw_options = [
        'Power tool', 'Aircraft', 'Tools', 'Helicopter', 'Drill'
    ]

    while True:
        
        detections = []

        recording = sd.rec(int(10 * 16000), samplerate=16000, channels=1, dtype='int16')
        sd.wait()
        wavwrite(audioFile, 16000, recording)

        base_options = python.BaseOptions(model_asset_path='models/yamnet.tflite')
        options = audio.AudioClassifierOptions(base_options=base_options, max_results=4)

        with audio.AudioClassifier.create_from_options(options) as classifier:
            sample_rate, wav_data = wavfile.read(audioFile)

            audio_clip = containers.AudioData.create_from_array(
                wav_data.astype(float) / np.iinfo(np.int16).max, sample_rate)
            
            classification_result_list = classifier.classify(audio_clip)

            for idx, timestamp in enumerate([0, 975, 1950, 2925]):
                classification_result = classification_result_list[idx]
                top_category = classification_result.classifications[0].categories[0]

                detections.append(top_category.category_name)

        if detections:
            if detections.count('chainsaws') >= 2:
                print('motosierra detectada')

        for p in possible_chainsaw_options:
            if detections.count(p) >= 2:
                print('posible motosierra detectada')
            

def gpt_send_message(prompt, image=None, model='gpt-5-nano'):
    message = [
        {
            'role': 'user',
            'content': [
                {'type': 'input_text', 'text': prompt}
            ]
        }
    ]

    if image:
        message[0]['content'].append({'type': 'input_image', 'image_url': f'data:image/png;base64,{image}'})

    response = client.responses.create(model=model, input=message)

    return response.output_text

def generate_fire_reports(lat, lon):
    last_fire_detection = time.time()
    reports = 0
    om = OpenMeteo(lat, lon)

    while True:
        if time.time() - last_fire_detection >= 3600 or reports == 0:
            wind_dir = requests.get(f'https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=wind_direction_10m').json()['current']['wind_direction_10m']

            sensors = get_sensor_data()

            datos_sensores = f'''
            Temperatura: {sensors['temperature']},
            Humedad: {sensors['humidity']},
            Velocidad viento: {om.get_wind_speed()},
            Direccion Viento: {wind_dir} Grados
            Latitud: {lat},
            Longitud: {lon}
            Direccion: {get_address(lat, lon)}
            '''


            response = gpt_send_message(
                PromptGPTReport + datos_sensores
            ) 

            
            last_fire_detection = time.time()
            reports += 1

            print(response)

def camera_detection_yolov8():
    global smoke_detected
    global fire_detected

    while True:
        smoke_detected = False
        fire_detected = False

        try:
            resp = requests.get(CameraURL, timeout=2)
            img_arr = np.frombuffer(resp.content, np.uint8)
            frame = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        except Exception as e:
            print("Error al obtener imagen:", e)
            continue

        results = model(frame, verbose=False, conf=0.25, iou=0.45)

        for r in results:
            for box in r.boxes:
                class_ = model.names[int(box.cls)].lower()
                
                if class_ in ['fire']:
                    fire_detected = True
                
                if class_ in ['smoke']:
                    smoke_detected = True

        cv2.imshow('Centinela', frame)

        key = cv2.waitKey(1) & 0xFF  
        if key == 27 or key == ord('q'):
            break

def rule_thirty_exec(lat, lon):
    om = OpenMeteo(lat, lon)

    while True:
            sensors = get_sensor_data()

            risk = rule_thirty(sensors['temperature'], sensors['humidity'], om.get_wind_speed())

            if risk == 0:
                print('[regra dos 30] riesgo de incendio maximo')

            elif risk == 1:
                print('[regra dos 30] riesgo de incendio medio')

            elif risk == 2:
                print('[regra dos 30] riesgo de incendio bajo')

            time.sleep(10)

def predict_fire(lat, lon):
    om = OpenMeteo(lat, lon)
    model = joblib.load('models/FirePrediction.pkl')
    while True:
        month = int(datetime.now().strftime('%m'))

        if month in [12, 1, 2]:
            season = 2
        elif month in [3, 4, 5]:
            season = 0

        elif month in [6, 7, 8]:
            season = 3

        elif month in [9, 10, 11]:
            season = 1

        temps = om.get_maxmin_temp()
        maxtemp = (temps[0] * 9/5) + 32
        mintemp = (temps[1] * 9/5) + 32
        windspeed = om.get_wind_speed() * 0.621371
        temprange = maxtemp - mintemp
        windtempratio = windspeed / maxtemp
        today = datetime.today().timetuple().tm_yday

        p = model.predict(
            [
                [om.get_precipitation(), maxtemp, mintemp, windspeed, 2025, temprange, windtempratio, 9, season, om.get_lagged_precipitation(), om.get_lagged_windspeed(), today]
            ]
        )

        if p == 0 or p == [0]:
            print('[ml model] No hay riesgo de incendio para hoy')

        else:
            print('[ml model] Hay riesgo de incendio para hoy')
        break

def recopile_nesterov_data(lat ,lon):

    while True:
        rows = []
        today = datetime.now().strftime('%d-%m')
        hour = datetime.now().strftime('%H:%M')    

        with open('data/nesterov.csv') as csvread:
            
            reader = csv.reader(csvread)

            for row in reader:
                rows.append(row[0])

            if rows[-1] != today and hour.startswith('15'):
                sensors = get_sensor_data()

                T15 = sensors['temperature']
                Tdew = float(dewpoint_from_relative_humidity(T15 * units.degC, sensors['humidity'] * units.percent).m)
                rain = OpenMeteo(lat, lon).get_precipitation()

                with open('data/nesterov.csv', 'a', newline='') as csvwrite:
                    writer = csv.writer(csvwrite)
                    writer.writerow([today, hour, T15, Tdew, rain])

        time.sleep(10)

def check_ppm_gases():
    while True:
        sensors = get_sensor_data()

        co = sensors['co']
        ch4 = sensors['ch4']
        co2 = sensors['co2']
        smoke = sensors['smoke']

        if 0 < smoke < 50:
            print('[humo] nivel bajo de humo')
        
        elif 50 <= smoke < 200:
            print('[humo] nivel moderado de humo')

        elif smoke > 200:
            print('[humo] niveles peligrosos de humo')

        if 0 < co < 50:
            print('[co] nivel bajo de monoxido de carbono')

        elif 50 <= co < 200:
            print('[co] nivel moderado de monoxido de carbono')

        elif co > 200:
            print('[co] niveles peligrosos de monoxido de carbono')

        if 0 < ch4 < 1000:
            print('[ch4] nivel bajo de metano')

        elif 1000 < ch4 < 5000:
            print('[ch4] nivel moderado de metano')

        elif ch4 > 5000:
            print('[ch4] niveles peligros de ch4')

        time.sleep(10)


class OpenMeteo:
    def __init__(self, lat ,lon):
        self.lat = lat
        self.lon = lon

    def get_wind_speed(self):
        response = requests.get(f'https://api.open-meteo.com/v1/forecast?latitude={self.lat}&longitude={self.lon}&current=wind_speed_10m').json()['current']['wind_speed_10m']

        return response

    def get_precipitation(self):
        response = requests.get(f'https://api.open-meteo.com/v1/forecast?latitude={self.lat}&longitude={self.lon}&current=precipitation').json()['current']['precipitation']

        return response
    
    def get_maxmin_temp(self):
        response = requests.get(f'https://api.open-meteo.com/v1/forecast?latitude={self.lat}&longitude={self.lon}&daily=temperature_2m_max,temperature_2m_min&timezone=auto').json()['daily']

        maxtemp = response['temperature_2m_max'][0]
        mintemp = response['temperature_2m_min'][0]

        return [maxtemp, mintemp]
    
    def get_lagged_precipitation(self):
        p = 0

        end_date = (datetime.today() - timedelta(days=1)).date()
        start_date = (end_date - timedelta(days=6)) 

        response = requests.get(f'https://archive-api.open-meteo.com/v1/archive?latitude={self.lat}&longitude={self.lon}&start_date={start_date.isoformat()}&end_date={end_date.isoformat()}&daily=precipitation_sum&timezone=auto').json()['daily']['precipitation_sum']

        for x in response:
            if x == 'null' or x == None:
                p += 0
            
            else:
                p += x

        return p
    
    def get_lagged_windspeed(self):
        prom = 0

        end_date = (datetime.today() - timedelta(days=1)).date()
        start_date = (end_date - timedelta(days=6)) 
    
        response = requests.get(f'https://archive-api.open-meteo.com/v1/archive?latitude={self.lat}&longitude={self.lon}&start_date={start_date.isoformat()}&end_date={end_date.isoformat()}&daily=windspeed_10m_max&timezone=auto').json()['daily']['windspeed_10m_max']

        for x in response:
            if x == 'null' or x == None:
                prom += 0
            
            else:
                prom += x

        promedio = prom / len(response)

        return promedio


class Indices:
    def __init__(self):
        pass

    def angstrom(self):
        """
        Indice de angstrom, las variables deben ser a las 13:00
        """

        sensors = get_sensor_data()

        I = (sensors['humidity']/20) + ((27 - sensors['temperature'])/10)


        if I > 4.0:
            print('[angström] Probabilidad baja de incendios')

        elif I > 2.5 and I <= 4.0:
            print('[angström] Condiciones de incendio desfavorables.')
        
        elif I > 2.0 and I <= 2.5:
            print('[angström] Condiciones de incendio favorables')
        
        elif I <= 2.0:
            print('[angström] Probabilidad muy alta de que se produzca un incendio.')

    
    def munger(self, w: int):
        Munger = 1/2 * w**2

        return Munger
    
    def nesterov_csv_today(self, lat, lon):
        while True:
            with open('data/nesterov.csv') as file:
                next(file)

                reader = csv.reader(file)

                NI = 0
                T15 = []
                Tdew = []
                rain = []
                indices = []

                for row in reader:
                    T15.append(float(row[2]))
                    Tdew.append(float(row[3]))
                    rain.append(float(row[4]))


                for i in range(len(T15)):
                    if rain[i] > 3:
                        NI = 0
                    else:
                        daily_value = (T15[i] - Tdew[i]) * T15[i]
                        NI += daily_value
                    
                    indices.append(NI)

                sensors = get_sensor_data()
                rain = OpenMeteo(lat, lon).get_precipitation()

                if rain >= 3:
                    NI = 0
                else:
                    T15 = sensors['temperature']
                    Tdew = float(dewpoint_from_relative_humidity(T15 * units.degC, sensors['humidity'] * units.percent).m)

                    today = (T15 - Tdew) * T15

                    NI += today

                if NI <= 300:
                    print('[nesterov] Sin peligro de incendio')

                elif NI <= 1000:
                    print('[nesterov] Peligro de incendio bajo')

                elif NI <= 4000:
                    print('[nesterov] Peligro de incendio medio')

                elif NI <= 10000:
                    print('[nesterov] Peligro de incendio ALTO')

                else:
                    print('[nesterov] PELIGRO DE INCENDIO EXTREMO')

                time.sleep(86400)
    
    def angstrom_exec(self):
        execs = 0
        while True:
            hour = datetime.now().time().hour

            #if hour == '13' and execs == 0:
            self.angstrom()

            #if hour != '13':
            #    execs = 0
            time.sleep(10)

class Main:

    def __init__(self):
        self.lat = '-34.98279'
        self.lon = '-71.23943'

    def init(self):
        Process(target=generate_fire_reports, args=(self.lat, self.lon)).start()
        #Process(target=camera_detection_yolov8).start()
        #Process(target=chainsaw_detect, args=('output.wav',)).start()
        Process(target=rule_thirty_exec, args=(self.lat, self.lon)).start()
        Process(target=Indices().angstrom_exec).start()
        Process(target=recopile_nesterov_data, args=(self.lat, self.lon)).start()
        Process(target=Indices().nesterov_csv_today, args=(self.lat, self.lon)).start()
        Process(target=predict_fire, args=(self.lat, self.lon)).start()
        Process(target=check_ppm_gases).start()

        print(banner)


    def __evaluar_regla30(self, temperature: int, humidity: int, wind_speed: int):
        risk = rule_thirty(temperature, humidity, wind_speed)

        if risk == 0:
            requests.post(ApiUrl + '/api/alert', json={'alert_status': 'max_alert'})

    def __generar_reporte(temperature: int, humidity: int, wind_speed: int, lat: str, lon: str, smoke: str, wind_dir: int):
        datos_sensores = f'''
Temperatura: {temperature},
Humedad: {humidity},
Velocidad viento: {wind_speed},
Direccion Viento: {wind_dir}
Latitud: {lat},
Longitud: {lon}
Direccion: {get_address(lat, lon)}
'''
        response = gpt_send_message(
            PromptGPTReport + datos_sensores
        )
        
        return json.loads(response)

if __name__ == "__main__":
    #print(Indices().angstrom(25, 56))
    Main().init()
    #Indices().nesterov_check('-34.98279', '-71.23943')
    #recopile_nesterov_data('-34.98279', '-71.23943')
    #print(Indices().nesterov_csv_today('-34.98279', '-71.23943'))