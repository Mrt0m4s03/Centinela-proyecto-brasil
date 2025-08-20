from openai import OpenAI
import requests
import base64


open_ai_apikey = ''
esp32_cam_url = 'http://10.19.71.100/capture'
esp32_sensor_url = ''
prompt_image = '''
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

client = OpenAI(api_key=open_ai_apikey)

def get_wind_speed(lat, lon):
    response = requests.get(f'https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=wind_speed_10m').json()['current']['wind_speed_10m']

    return response

def detect_fire_risk(temp, humidity, wind):
    """
    Detecta el riesgo de incendio usando la regla 30 30 30
    """
    if temp >= 30 and humidity <= 30 and wind >= 30:
        return 2
    
    if temp >= 25 and humidity <= 35 and wind >= 25:
        return 1
    
    else: 
        return 0
    
def send_openai_message(prompt, img=None, model='gpt-5-nano'):
    content = [
        {'type': 'input_text', 'text': prompt}
    ]

    if img:
        content.append({'type': 'input_image', 'image_url': f'data:image/png;base64,{img}'})

    response = client.responses.create(
        model=model,
        input=[
            {
                'role': 'user',
                'content': content
            }
        ]
    )

    print(response.output_text)

while True:
    image = base64.b64encode(requests.get(esp32_cam_url).content).decode('utf-8')

    response = send_openai_message(prompt_image, image)
