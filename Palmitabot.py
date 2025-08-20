import network
import socket
import machine
import bme280
import ssd1306
import time
from time import sleep
from machine import Pin
import requests
import _thread

# Codigo control remoto basado en https://www.youtube.com/watch?v=RonINjKL-I0

# Configuración de la conexión con Blynk para publicar los datos de los sensores
BLYNK_AUTH_TOKEN = "YmvfgqwZIcxQseiFCu27I27FRgy4mjpu"

################## - RED - ##########################
# Configuración de la conexión Wi-Fi #

ssid = 'S20FE'
password = '11223343'

def conectar():
    red = network.WLAN(network.STA_IF)
    red.active(True)
    red.connect(ssid, password)
    while red.isconnected() == False:
        print('Conectando a la red Wi-Fi...')
        sleep(1)
    ip = red.ifconfig()[0]
    print(f'Conectado con IP: {ip}')
    return ip


################## - MOTORES - ######################
# Configuración de los pines GPIO para el control de los motores

Mot_A_Forward = Pin(2, Pin.OUT)
Mot_A_Back = Pin(3, Pin.OUT)
Mot_B_Forward = Pin(4, Pin.OUT)
Mot_B_Back = Pin(5, Pin.OUT)

def move_forward():
    Mot_A_Forward.value(1)
    Mot_B_Forward.value(1)
    Mot_A_Back.value(0)
    Mot_B_Back.value(0)
    
def move_backward():
    Mot_A_Forward.value(0)
    Mot_B_Forward.value(0)
    Mot_A_Back.value(1)
    Mot_B_Back.value(1)

def move_stop():
    Mot_A_Forward.value(0)
    Mot_B_Forward.value(0)
    Mot_A_Back.value(0)
    Mot_B_Back.value(0)

def move_left():
    Mot_A_Forward.value(1)
    Mot_B_Forward.value(0)
    Mot_A_Back.value(0)
    Mot_B_Back.value(1)

def move_right():
    Mot_A_Forward.value(0)
    Mot_B_Forward.value(1)
    Mot_A_Back.value(1)
    Mot_B_Back.value(0)

move_stop()
sleep(2)



##           Testeo de motores          ##

print("Forward test")
move_forward()
sleep(1)

print("Backward test")
move_backward()
sleep(1)

print("Spin left test")
move_left()
sleep(1)

print("Spin right test")
move_right()
sleep(1)

print("'Time for bed' said Zeberdee.")
move_stop()

##########################################

##########################################
##        Configuración pantalla        ##
i2c = machine.I2C(0, sda=machine.Pin(0), scl=machine.Pin(1))
oled = ssd1306.SSD1306_I2C(128, 32, i2c)

# Mostrar texto inicial en la pantalla OLED
oled.fill(0)
oled.text("Hola, mi nombre", 0, 0)
oled.text("es PALMA Bot", 0, 10)
oled.show()
time.sleep(2)

oled.fill(0)
oled.text("Soy una estacion", 0, 0)
oled.text("de monitoreo", 0, 10)
oled.text("ambiental", 0, 20)
oled.show()
time.sleep(2)

oled.fill(0)
oled.text("A continuacion", 0, 0)
oled.text("te mostrare la", 0, 10)
oled.text("calidad del", 0, 20)
oled.text("Ambiente", 0, 30)
oled.show()
time.sleep(2)

oled.fill(0)
oled.text("AMBIENTE", 40, 10)
oled.text("{-------}", 33, 20)
oled.show()
time.sleep(2)  # Esperar 3 segundos antes de mostrar los datos del sensor

################## - SENSORES - ######################
def write(token,pin,value):
    api_url = "https://blynk.cloud/external/api/update?token="+token+"&"+pin+"="+value
    response = requests.get(api_url)
    if "200" in str(response):
        print("Valor actualizado exitosamente")
    else:
        print("No se puede encontrar el dispositivo (token incorrecto) o el formato del pin es incorrecto")
    response.close()

def sensores(BLYNK_AUTH_TOKEN):
    c = 0
    while(c<6):
       
        
        # Lectura analógica del sensor MQ-135
        analog_pin = machine.ADC(26)  # Pin analógico del sensor MQ-135
        analog_value = analog_pin.read_u16()  # Lectura en formato de 16 bits

        # Cálculo de la concentración de gases
        voltage = analog_value * 3.3 / 65535  # Conversión a voltaje

        # Fórmulas de conversión para diferentes gases
        co2_ppm = 5000 * voltage / 3.3  # Concentración de CO2 en partes por millón (ppm)
        co_ppm = 50 * voltage / 3.3  # Concentración de CO en partes por millón (ppm)
        nox_ppm = 100 * voltage / 3.3  # Concentración de NOx en partes por millón (ppm)
        
        
        try:
            # Imprimir los valores de concentración de gases del MQ-135
            print("Concentración de CO2: {:.2f} ppm".format(co2_ppm))
            # Postear valores en Blynk para almacenar un historial ambiental
            write(BLYNK_AUTH_TOKEN,"v1", "{}".format(co2_ppm))
        except:
            print("error publicando CO2")
            
        try:
            print("Concentración de CO: {:.2f} ppm".format(co_ppm))
            write(BLYNK_AUTH_TOKEN,"v2", "{}".format(co_ppm))
        except:
            print("error publicando CO")
            
        try:
            print("Concentración de NOx: {:.2f} ppm".format(nox_ppm))
            write(BLYNK_AUTH_TOKEN,"v3", "{}".format(nox_ppm))
        except:
            print("error publicando NOx")
        time.sleep(10)
        c = c+1
        
# Función para abrir el puerto 80 y esperar una petición HTTP

def open_socket(ip):
    address = (ip, 80)
    connection = socket.socket()
    connection.bind(address)
    connection.listen(1)
    return connection

# Función para crear una página web con los botones de control del robot

def pagina_web():
    html = f"""
            <!DOCTYPE html>
            <html>
            <head>
            </head>
            <body>
            <center>
            <form action="./adelante">
            <input type="submit" value="Adelante" style="background-color: #04AA6D; border-radius: 15px; height:120px; width:120px; border: none; color: white; padding: 16px 24px; margin: 4px 2px"  />
            </form>
            <table><tr>
            <td><form action="./izquierda">
            <input type="submit" value="Izquierda" style="background-color: #04AA6D; border-radius: 15px; height:120px; width:120px; border: none; color: white; padding: 16px 24px; margin: 4px 2px"/>
            </form></td>
            <td><form action="./detener">
            <input type="submit" value="Detener" style="background-color: #FF0000; border-radius: 50px; height:120px; width:120px; border: none; color: white; padding: 16px 24px; margin: 4px 2px" />
            </form></td>
            <td><form action="./derecha">
            <input type="submit" value="Derecha" style="background-color: #04AA6D; border-radius: 15px; height:120px; width:120px; border: none; color: white; padding: 16px 24px; margin: 4px 2px"/>
            </form></td>
            </tr></table>
            <form action="./atras">
            <input type="submit" value="Atras" style="background-color: #04AA6D; border-radius: 15px; height:120px; width:120px; border: none; color: white; padding: 16px 24px; margin: 4px 2px"/>
            </form>
            <form action="./medir">
            <input type="submit" value="Medir" style="background-color: blue; border-radius: 15px; height:120px; width:120px; border: none; color: white; padding: 16px 24px; margin: 8px 8px"/>
            </form>
            </body>
            </html>
            """
    return str(html)

def serve(connection, BLINK_AUTH_TOKEN):
    print("Iniciando servidor web")
    
    while True:
        cliente = connection.accept()[0]
        peticion = cliente.recv(1024)
        peticion = str(peticion)
                
        try:
            peticion = peticion.split()[1]
        except IndexError:
            pass
        
        if peticion == '/adelante?':
            move_forward()
            
        elif peticion =='/izquierda?':
            move_left()
            
        elif peticion =='/detener?':
            
            move_stop()
            
        elif peticion =='/derecha?':
            move_right()
            
        elif peticion =='/atras?':
            move_backward()
            
        elif peticion =='/medir?':
            
            sensores(BLYNK_AUTH_TOKEN)
        html = pagina_web()
        cliente.send(html)
        cliente.close()




try:
    ip = conectar()
    connection = open_socket(ip)
    serve(connection, sensores)
    
except KeyboardInterrupt:
    machine.reset()