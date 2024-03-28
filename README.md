# Repositorio para Algoritmo de Vision por Computador para el proyecto NeuroCogTrack

## Para el funcionamiento de este proceso, se manejan dos carpetas, las cuales cumplen diferentes roles:
### mcp-vision-api:
En esta carpeta, se maneja un api la cual recibe las peticiones del frontend y llama los metodos de los diferentes algoritmo para hacer la conexión y enviarle los parametros necesarios.

#### Rutas API
##### Ruta de Inicio

- **Ruta:** `/`
- **Método:** `GET`
- **Descripción:** Da la bienvenida al usuario a la aplicación (Esto con fines de prueba).

##### Ruta para Crear Imagen

- **Ruta:** `/create-image`
- **Método:** `GET`
- **Descripción:** Genera y envía un archivo de imagen al cliente (Esto con fines de prueba).

##### Ruta de Calibración Automática

- **Ruta:** `/calibration_automatic`
- **Método:** `POST`
- **Parámetros de Entrada:** `Screenshot` (Imagen background para análisis)
- **Descripción:** Realiza la calibración automática basándose en una captura de pantalla proporcionada.

##### Ruta de Calibración Semi-Automática

- **Ruta:** `/calibration_semiautomatic`
- **Método:** `POST`
- **Parámetros de Entrada:** `Screenshot` (Imagen background para análisis), `marks` (Marcas otorgadas por el usuario)
- **Descripción:** Realiza la calibración semi-automática usando una captura de pantalla y las marcas de calibración proporcionadas.

##### Ruta de Análisis Automático

- **Ruta:** `/autoAnalysis`
- **Método:** `POST`
- **Parámetros de Entrada:** `contourjson` (output de la calibración), `videoUrl` (URL del video en el S3), `imageUrl` (URL de la imagen background en el S3), `jsonString` (Lista con los estimulos de la sesión)
- **Descripción:** Realiza un análisis automático basado en las entradas proporcionadas y devuelve los resultados del análisis.

### mcp-vision-detection:
  En esta carpeta, se tienen todos los archivos que realizan los diferentes procesos necesarios para analisis, se exportan los modulos del algoritmo y se utiliza un archivo binding.gyp para hacer la conexión entre javascript (API) y los modulos de c++.
Podemos agrupar los archivos dentro de src/lib/* en dos categorias, en donce:
- Calibration_fixed: Se encarga de la deteccion de marcas, utilizando como herramientas ConvertImage y CommonDefitions.
- ComputerVisionWeb: Se encarga del trackeo de pies y la generación del output para el análisis del operador, usa como herramientas los archivos ConvertImage, FeetTracker, ExtendedContour y CommonDefitions.
- CommonDefitions: Archivo el cual realiza las importaciones necesarias y define las estructuras para diferentes usos.


### Comandos utiles:
- docker build -t vision .
- docker run -p 3001:3001 vision
