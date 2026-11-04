# Stage 1: Build the Flutter Web App
FROM ghcr.io/cirruslabs/flutter:3.35.6 AS build-stage
WORKDIR /app

# Get project dependencies
COPY src/pubspec.* .
RUN flutter pub get

# Copy the rest of the source code
COPY src .

# Immagine unica per tutti i siti: nome, archivio e home page arrivano dal
# backend in base al dominio. Senza service worker: i siti cambiano senza un
# nuovo deploy e una copia in cache non deve mostrarne una versione vecchia.
RUN flutter clean
RUN flutter build web --release --pwa-strategy=none

# --------------------------------------------------------------------------

# Stage 2: Serve the app with Nginx (Production Stage)
FROM nginx:alpine

# Copy custom Nginx configuration (optional, for SPA routing)
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copy the built Flutter files from the build stage into the Nginx content directory
COPY --from=build-stage /app/build/web /usr/share/nginx/html
EXPOSE 80