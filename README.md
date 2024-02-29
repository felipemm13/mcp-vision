# mcp-vision
Repositorio para Algoritmo de deteccion de marcas proyecto NeuroCogTrack

Para el funcionamiento de este, se manejan dos carpetas:
- mcp-vision-api
  En esta carpeta, se maneja un api la cual recibe las peticiones del frontend y llama los metodos del algoritmo de detección de marcas para hacer la conexión.
- mcp-vision-detection
  En esta carpeta, se exportan los modulos del algoritmo, se utiliza un archivo binding.gyp para hacer la conexión entre javascript y los modulos de c++.

### Comandos utiles:
- docker build -t vision .
- docker run -p 3001:3001 vision
- node-gyp rebuild