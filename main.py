import time
import json
import cv2
import requests
import sys
import numpy as np
from ultralytics import YOLO
from datetime import datetime
from colorama import init, Fore, Back, Style
from geopy.geocoders import Nominatim
from openai import OpenAI

init(autoreset=True)

OPENAI_MODEL = 'gpt-5-nano'
API_URL = 'http://10.19.71.68:1337'
ESP_32_CAM_URL = 'http://10.19.71.100/capture'

model = YOLO('models/FireAndSmoke.pt')

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

def get_kmph(lat, lon):
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=wind_speed_10m"
    response = requests.get(url)
    data = response.json()
    wind_speed = data['hourly']['wind_speed_10m'][0]
    return wind_speed

def detect_fire_risk(humidity, temperature, wind_speed):
    if humidity < 30 and temperature > 30 and wind_speed > 30:
        return 2
    if humidity < 35 and temperature > 25 and wind_speed > 25:
        return 1
    else:
        return 0

def get_address(lat, lon):
    geolocator = Nominatim(user_agent="Centinela/1.0.0")
    location = geolocator.reverse((lat, lon), language='es')
    if location:
        return location.address
    else:
        return None
    
def fire_report(nombre_ubicacion, latitud, longitud, temperatura_celsius, humedad_porcentaje, viento_kmh, humo_detectado, fecha_hora=None):
    prompt = '''
Eres un asistente experto en análisis ambiental para detección temprana de incendios forestales, busca en internet informacion climatica, geografica, vegetacion, y mas cosas para mejorar el informe, el umbral del MQ 2, son entre 700-+1000, eso significa que eso no podria sobrepasar.,
Recomienda buenas acciones y recomendaciones, no cualquiera, las necesarias, si no se requieren acciones o recomendaciones no pongas nada, solo pon lo necesario segun las condiciones que te pasare, te recalco que solo digas lo justo y necesario.
Te entregaré datos de sensores y condiciones meteorológicas para evaluar el riesgo según la regla 30-30-30 (temperatura > 30°C, humedad < 30%, viento > 30 km/h). Debes generar un informe en formato JSON con la siguiente estructura:

{
  "fecha_hora": "string con fecha y hora ISO 8601",
  "ubicacion": {
    "nombre": "nombre del lugar",
    "latitud": número decimal,
    "longitud": número decimal
  },
  "datos_sensores": {
    "temperatura_celsius": número,
    "humedad_porcentaje": número,
    "viento_kmh": número,
    "humo_detectado": número
  },
  "evaluacion_regla_30_30_30": {
    "cumple_regla": booleano,
    "descripcion_estado": "texto explicativo del estado"
  },
  "informe_texto": "un resumen claro y en texto plano sobre las condiciones y riesgo",
  "recomendaciones": [
    "lista de recomendaciones concretas"
  ],
  "acciones_sugeridas": [
    {
      "accion": "acción específica a realizar",
      "prioridad": "alta/media/baja"
    }
}'''+f'''

Datos de entrada:
Fecha y hora: {fecha_hora}
Lugar: {nombre_ubicacion}, latitud {latitud}, longitud {longitud}
Temperatura: {temperatura_celsius} °C
  ],
  "estado_alarma": "activada/desactivada",
  "notas_adicionales": "cualquier otra información útil"
Humedad: {humedad_porcentaje} %
Viento: {viento_kmh} km/h
Humo Sensor MQ-2 Analogico: {humo_detectado}

Genera la respuesta JSON sin explicaciones adicionales.

'''
    
    #geolocator = Nominatim(user_agent='Centinela/1.0.0')
    #location = geolocator.reverse((latitud, longitud))

    client = OpenAI(
        api_key=''
    )

    completion = client.chat.completions.create(
        model=OPENAI_MODEL, # gpt-4o-mini-search-preview
        #web_search_options={
        #    'user_location': {
        #        'type': 'approximate',
        #        'approximate': {
        #            'country': 'CL',
        #            'city': location.raw['address']['city'],
        #            'region': location.raw['address']['state'],
        #        }
        #    }
        #}, 
        messages=[
            {
                'role': 'user',
                'content': prompt
            }
        ]
    )

    return completion.choices[0].message.content

def get_sensor_data():
    response = requests.get('http://10.19.71.68:1337/api/sensor_data')


    return json.loads(response.text)[0]



def main(lat, lon):
    reports = 0
    last_report = time.time()
    last_report_data = None
    last_kmph_request = time.time()
    last_fire_detection = time.time()
    kmph_value = None
    last_status = None

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(banner)
    print(f"[{now}] {Back.BLUE}{Fore.WHITE}ESP32 CAM{Style.RESET_ALL} Iniciando camara")

    while True:
        smoke_detected = False
        fire_detected = False
        fire_and_smoke_detected = False

        try:
            resp = requests.get(ESP_32_CAM_URL, timeout=2)
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


        # Mostrar la imagen con OpenCV
        cv2.imshow('Centinela', frame)

        # Salir con 'q' o ESC
        key = cv2.waitKey(1) & 0xFF  
        if key == 27 or key == ord('q'):
            break

        # Pequeña espera para no saturar la cámara
        time.sleep(0.1)

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if time.time() - last_fire_detection >= 30:
            if fire_and_smoke_detected:
                print(f"[{now}] {Back.BLUE}{Fore.WHITE}ESP32 CAM{Style.RESET_ALL} FUEGO Y HUMO {Back.RED}DETECTADO{Style.RESET_ALL}")
                requests.post(API_URL + '/api/alert', json={'ai_status': 'fire_and_smoke_detected'})

            if fire_detected:
                print(f"[{now}] {Back.BLUE}{Fore.WHITE}ESP32 CAM{Style.RESET_ALL} FUEGO {Back.RED}DETECTADO{Style.RESET_ALL}")
                requests.post(API_URL + '/api/alert', json={'ai_status': 'fire_detected'})

            if smoke_detected:
                print(f"[{now}] {Back.BLUE}{Fore.WHITE}ESP32 CAM{Style.RESET_ALL} Humo {Back.YELLOW}Detectado{Style.RESET_ALL}")
                requests.post(API_URL + '/api/alert', json={'ai_status': 'smoke_detected'})

            last_fire_detection = time.time()
            

        
        sensors = get_sensor_data()

        if time.time() - last_kmph_request >= 30:
            last_kmph_request = time.time()

            fire_risk = detect_fire_risk(
                sensors['humidity'],
                sensors['temperature'],
                get_kmph(lat, lon)
            )

            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            if fire_risk == 0:
                print(f"[{now}] {Back.LIGHTYELLOW_EX}{Fore.BLACK}FIRE RISK{Style.RESET_ALL}{Style.RESET_ALL} {Fore.GREEN}No hay alerta{Fore.WHITE} de incendio{Style.RESET_ALL}")
                requests.post(API_URL + '/api/alert', json={'alert_status': 'max_alert'})


            if fire_risk == 1:
                print(f"[{now}] {Back.LIGHTYELLOW_EX}{Fore.WHITE}FIRE RISK{Style.RESET_ALL}{Style.RESET_ALL} {Fore.YELLOW}Alerta{Fore.WHITE}: Condiciones propensas a incendios{Style.RESET_ALL}")
                requests.post(API_URL + '/api/alert', json={'alert_status': 'medium_alert'})

            if fire_risk == 2:
                print(f"[{now}] {Back.RED}{Fore.WHITE}FIRE RISK{Style.RESET_ALL}{Style.RESET_ALL} {Back.RED}{Fore.WHITE}¡Peligro! Condiciones extremas para incendios{Style.RESET_ALL}")
                requests.post(API_URL + '/api/alert', json={'alert_status': 'low_alert'})

            last_kmph_request = time.time()

        if reports == 0 or (time.time() - last_report >= 3600):
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f'[{now}] {Back.WHITE}{Fore.BLACK}{OPENAI_MODEL}{Style.RESET_ALL} Generando informe...')

            report = json.loads(fire_report(
                get_address(lat, lon),
                lat,
                lon,
                sensors['temperature'],
                sensors['humidity'],
                get_kmph(lat, lon),
                sensors['smoke'],
                now
            ))

            requests.post(API_URL + '/api/report', json=report)

            print(f'[{now}] {Back.WHITE}{Fore.BLACK}{OPENAI_MODEL}{Style.RESET_ALL} Descripcion informe: {report['informe_texto']}')
            print(f'[{now}] {Back.WHITE}{Fore.BLACK}{OPENAI_MODEL}{Style.RESET_ALL} Descripcion 30-30-30: {report['evaluacion_regla_30_30_30']['descripcion_estado']}')
            print(f'[{now}] {Back.WHITE}{Fore.BLACK}{OPENAI_MODEL}{Style.RESET_ALL} Total de recomendaciones: {len(report['recomendaciones'])}')
            print(f'[{now}] {Back.WHITE}{Fore.BLACK}{OPENAI_MODEL}{Style.RESET_ALL} Total de acciones sugeridas: {len(report['acciones_sugeridas'])}')

            last_report = time.time()
            reports += 1

        

main(-34.851144, -72.14149)