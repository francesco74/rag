# Stage 1: Build the Flutter Web App
FROM ghcr.io/cirruslabs/flutter:3.35.6 AS build-stage
WORKDIR /app

# Get project dependencies
COPY src/pubspec.* .
RUN flutter pub get

# Copy the rest of the source code
COPY src .

# Build the web app, passing Docker ARGs to Dart's --dart-define
RUN flutter clean
RUN flutter build web --release 

# --------------------------------------------------------------------------

# Stage 2: Serve the app with Nginx (Production Stage)
FROM nginx:alpine

# Copy custom Nginx configuration (optional, for SPA routing)
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copy the built Flutter files from the build stage into the Nginx content directory
COPY --from=build-stage /app/build/web /usr/share/nginx/html

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
