FROM postgis/postgis:16-3.4

# Instala unzip para poder descomprimir archivos .zip
RUN apt-get update \
    && apt-get install -y --no-install-recommends unzip \
    && rm -rf /var/lib/apt/lists/*

# Copiamos el script que restaurará el dump cuando el contenedor se inicie
COPY docker/init-db.sh /docker-entrypoint-initdb.d/00-init-db.sh
RUN chmod +x /docker-entrypoint-initdb.d/00-init-db.sh

# Copiamos el dump dentro de la imagen para que se restaure automáticamente
COPY db/sea_crawler_several_dbs-2025_07_29_10_48_51-dump.sql.zip /docker-entrypoint-initdb.d/dump.sql.zip
