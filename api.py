from flask import Flask, jsonify, request
from threading import Lock

app = Flask(__name__)

sensor_data_store = []
alert_data_store = []
report_data_store = []
data_lock = Lock()

@app.route('/api/esp32_upload_data', methods=['POST'])
def esp32_upload_data():
    data = request.json

    if not data:
        return jsonify({"error": "No JSON data received"}), 400
        
    required_fields = ["humidity", "temperature", "smoke"]

    for field in required_fields:
        if field not in data:
            return jsonify({"error": f"Missing field: {field}"}), 400

    with data_lock:
        sensor_data_store.append(data)

    return jsonify({"message": "Data received successfully"}), 200

@app.route('/api/alert', methods=['POST', 'GET'])
def alert():

    if request.method == 'POST':
        alert_data_store.clear()
        data = request.get_json(force=False, silent=True)
        if not data:
            return jsonify({"error": "No JSON data received"}), 400
        
        required_fields = ['alert_status']

        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing field: {field}"}), 400
            
        with data_lock:
            alert_data_store.append(data)

        return jsonify({"message": "Alert received successfully"}), 200

    if request.method == 'GET':
        with data_lock:
            return jsonify(alert_data_store), 200
        
@app.route('/api/report', methods=['POST', 'GET'])
def report():

    if request.method == 'POST':
        data = request.get_json(force=False, silent=True)
        if not data:
            return jsonify({"error": "No JSON data received"}), 400
        
        #required_fields = ['report_data',]

        #for field in required_fields:
        #    if field not in data:
        #        return jsonify({"error": f"Missing field: {field}"}), 400
            
        with data_lock:
            report_data_store.append(data)

        return jsonify({"message": "Alert received successfully"}), 200

    if request.method == 'GET':
        with data_lock:
            return jsonify(report_data_store), 200
            


@app.route('/api/sensor_data', methods=['GET'])
def get_sensor_data():
    with data_lock:
        return jsonify(sensor_data_store), 200

if __name__ == '__main__':
    app.run(
        debug=True,
        host='0.0.0.0',
        port=1337
    )