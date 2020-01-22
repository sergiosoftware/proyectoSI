# Importar lo relacionado con el framework flask para los servicios
import numpy as np
from flask import Flask, render_template, request, jsonify
from flask_json import FlaskJSON, JsonError, json_response, as_json, json
from flask_cors import CORS, cross_origin
from datetime import datetime
from prediccion import prediccion
import cv2
from array import *
# Codificar y decodificar la imagen que se va a analizar
import base64

# Instancia de flask
app = Flask(__name__)
FlaskJSON(app)
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
# Definir el route con el metodo POST


@app.route('/reconocimiento',methods=['POST', 'OPTIONS'])
@cross_origin(allow_headers=['Content-Type'], origin='*')
def reconocimiento():
    #image_64_decode = base64.b64decode("data:image/jpeg;base64,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dEJvdHRvbWxvbmcAAAAAAAAADGNyb3BSZWN0TGVmdGxvbmcAAAAAAAAADWNyb3BSZWN0UmlnaHRsb25nAAAAAAAAAAtjcm9wUmVjdFRvcGxvbmcAAAAAADhCSU0D7QAAAAAAEABIAAAAAQACAEgAAAABAAI4QklNBCYAAAAAAA4AAAAAAAAAAAAAP4AAADhCSU0EDQAAAAAABAAAAB44QklNBBkAAAAAAAQAAAAeOEJJTQPzAAAAAAAJAAAAAAAAAAABADhCSU0nEAAAAAAACgABAAAAAAAAAAI4QklNA/UAAAAAAEgAL2ZmAAEAbGZmAAYAAAAAAAEAL2ZmAAEAoZmaAAYAAAAAAAEAMgAAAAEAWgAAAAYAAAAAAAEANQAAAAEALQAAAAYAAAAAAAE4QklNA/gAAAAAAHAAAP////////////////////////////8D6AAAAAD/////////////////////////////A+gAAAAA/////////////////////////////wPoAAAAAP////////////////////////////8D6AAAOEJJTQQIAAAAAAAQAAAAAQAAAkAAAAJAAAAAADhCSU0EHgAAAAAABAAAAAA4QklNBBoAAAAAA0UAAAAGAAAAAAAAAAAAAACAAAAAgAAAAAgAMgA5ADYAMAA4ADQAMAAxAAAAAQAAAAAAAAAAAAAAAAAAAAAAAAABAAAAAAAAAAAAAACAAAAAgAAAAAAAAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAAAAAAAAAABAAAAABAAAAAAAAbnVsbAAAAAIAAAAGYm91bmRzT2JqYwAAAAEAAAAAAABSY3QxAAAABAAAAABUb3AgbG9uZwAAAAAAAAAATGVmdGxvbmcAAAAAAAAAAEJ0b21sb25nAAAAgAAAAABSZ2h0bG9uZwAAAIAAAAAGc2xpY2VzVmxMcwAAAAFPYmpjAAAAAQAAAAAABXNsaWNlAAAAEgAAAAdzbGljZUlEbG9uZwAAAAAAAAAHZ3JvdXBJRGxvbmcAAAAAAAAABm9yaWdpbmVudW0AAAAMRVNsaWNlT3JpZ2luAAAADWF1dG9HZW5lcmF0ZWQAAAAAVHlwZWVudW0AAAAKRVNsaWNlVHlwZQAAAABJbWcgAAAABmJvdW5kc09iamMAAAABAAAAAAAAUmN0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RTJDRDI3M0FFMjNGQzdEREM1RTc4NDM3QzMiIHhtcE1NOkluc3RhbmNlSUQ9InhtcC5paWQ6ZGU1OTRlOTYtYTE3Mi00NWQwLWE4MjEtZDZiNzBjMTc5NmYyIiB4bXBNTTpPcmlnaW5hbERvY3VtZW50SUQ9IjAyRTVGRkUyQ0QyNzNBRTIzRkM3RERDNUU3ODQzN0MzIiBkYzpmb3JtYXQ9ImltYWdlL2pwZWciIHBob3Rvc2hvcDpDb2xvck1vZGU9IjMiIHBob3Rvc2hvcDpJQ0NQcm9maWxlPSIiIHhtcDpDcmVhdGVEYXRlPSIyMDE5LTEyLTExVDEzOjEwOjE4LTA1OjAwIiB4bXA6TW9kaWZ5RGF0ZT0iMjAxOS0xMi0xMVQxMzoxNTowNi0wNTowMCIgeG1wOk1ldGFkYXRhRGF0ZT0iMjAxOS0xMi0xMVQxMzoxNTowNi0wNTowMCI+IDx4bXBNTTpIaXN0b3J5PiA8cmRmOlNlcT4gPHJkZjpsaSBzdEV2dDphY3Rpb249InNhdmVkIiBzdEV2dDppbnN0YW5jZUlEPSJ4bXAuaWlkOmRlNTk0ZTk2LWExNzItNDVkMC1hODIxLWQ2YjcwYzE3OTZmMiIgc3RFdnQ6d2hlbj0iMjAxOS0xMi0xMVQxMzoxNTowNi0wNTowMCIgc3RFdnQ6c29mdHdhcmVBZ2VudD0iQWRvYmUgUGhvdG9zaG9wIDIxLjAgKE1hY2ludG9zaCkiIHN0RXZ0OmNoYW5nZWQ9Ii8iLz4gPC9yZGY6U2VxPiA8L3htcE1NOkhpc3Rvcnk+IDwvcmRmOkRlc2NyaXB0aW9uPiA8L3JkZjpSREY+IDwveDp4bXBtZXRhPiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAg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")
    # Obtener la informacion enviada en los paramteros
    imagenAnalizar = request.get_json()
    print("Data recibida ",imagenAnalizar['imagen'])
    categorias = ["Gaviota occidental", "Gallinas", "Coragyps atratus (Gallinazo)", "Tacuarita", "Mirlo de alas rojas",
                  "green violetear", "Cardenal norteño", "Hapalopsittaca amazonina", "scarlet tanager",
                  "momoto amazónico (barranquillo coronado)", "Yellow headed Blackbird(Mirlo de cabeza amarilla)",
                  "Lazuli Bunting"]
    reconocimiento = prediccion()
    #imagenPrueba = cv2.imread("test/4/4_4_15.jpg", 0)
    img = base64.b64decode(imagenAnalizar['imagen'])
    npimg = np.fromstring(img, dtype=np.uint8)
    predicciones = reconocimiento.predecir(npimg)
    claseMayorValor = np.argmax(predicciones, axis=1)
    print("Estas son las probabilidades ", predicciones[0])
    print("La imagen cargada es ", categorias[claseMayorValor[0] - 1])
    # Retornar la respuesta
    # return "<h1>Bienvenido " + imagenAnalizar + "</h1>"
    # my_array = array('i', [1, 2, 3, 4])
    response = Flask.Response(jsonify(idImagen='img123', prediccion=categorias[claseMayorValor[0] - 1], probabilidades=json.dumps(str(predicciones[0]))))
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    response.headers['Content-Type'] = 'application/json'
    return response
    #return "<h1>Bienvenido</h1>"


@app.route('/')
def index():
    return "<h1>Bienvenido</h1>"


if __name__ == '__main__':
    # Iniciar la aplicacion
    app.run(debug=True)
