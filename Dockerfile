# Usar una imagen base de Node.js que sea más ligera, como la versión 'slim'
FROM node:14-slim

# Actualizar la lista de paquetes e instalar dependencias necesarias
RUN apt-get update && apt-get install -y \
    python \
    make \
    g++ \
    cmake \
    pkg-config \
    libopencv-dev \
    libcurl4-openssl-dev \
    nlohmann-json3-dev \
    libjsoncpp-dev \
    && rm -rf /var/lib/apt/lists/*

# Establecer el directorio de trabajo en el contenedor
WORKDIR /usr/src/app

# Copiar el package.json y package-lock.json (si aplica) para ambos proyectos
COPY mcp-vision-api/package*.json ./mcp-vision-api/
COPY mcp-vision-detection/package*.json ./mcp-vision-detection/

# Instalar las dependencias de Node.js para la API
RUN cd mcp-vision-api && npm install --only=production

# Instalar las dependencias de Node.js para el módulo de detección
RUN cd mcp-vision-detection && npm install

# Copiar el package.json y package-lock.json (si aplica) para ambos proyectos
COPY mcp-vision-api/package*.json ./mcp-vision-api/
COPY mcp-vision-detection/package*.json ./mcp-vision-detection/

# Instalar las dependencias de Node.js para la API
RUN cd mcp-vision-api && npm install --only=production

# Instalar las dependencias de Node.js para el módulo de detección
RUN cd mcp-vision-detection && npm install

# Copiar el resto de los archivos de la API y del módulo de detección al contenedor
COPY mcp-vision-api/ ./mcp-vision-api/
COPY mcp-vision-detection/ ./mcp-vision-detection/

# Compilar el módulo nativo. Este paso asume que tu package.json en mcp-vision-detection
# tiene un script "build" que llama a "node-gyp rebuild" o una operación similar.
RUN cd mcp-vision-detection && npm run build

# Exponer el puerto que la API utiliza
EXPOSE 3001

# Definir el comando para ejecutar la aplicación
CMD [ "node", "mcp-vision-api/index.js" ]