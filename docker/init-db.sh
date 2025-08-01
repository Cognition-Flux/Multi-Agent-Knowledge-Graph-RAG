#!/usr/bin/env bash
set -euo pipefail

# Variables clave proporcionadas por la imagen oficial de Postgres
: "${POSTGRES_DB:?Need to set POSTGRES_DB}"
: "${POSTGRES_USER:?Need to set POSTGRES_USER}"

# Usuario / rol que el dump espera encontrar
DUMP_ROLE="sea_crawler"
DUMP_ROLE_PWD="sea_crawler_pwd"

# -----------------------------------------------------------------------------------
# 1) Crear el rol que el dump requiere (si aún no existe) y asignarlo como dueño
# -----------------------------------------------------------------------------------

echo "[init-db] Creando rol ${DUMP_ROLE} (si no existe) y reasignando propietario..."
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname postgres <<-EOSQL
-- Crear rol requerido por el dump si no existe
DO \$\$
BEGIN
   IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = '${DUMP_ROLE}') THEN
      CREATE ROLE ${DUMP_ROLE} WITH LOGIN PASSWORD '${DUMP_ROLE_PWD}' SUPERUSER;
   END IF;
END\$\$;

-- Asegurar que también exista el rol "postgres" que el dump menciona (por si la imagen lo omitió)
DO \$\$
BEGIN
   IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'postgres') THEN
      CREATE ROLE postgres WITH LOGIN SUPERUSER;
   END IF;
END\$\$;

-- Transferir propiedad de la BD
ALTER DATABASE ${POSTGRES_DB} OWNER TO ${DUMP_ROLE};
EOSQL

# -----------------------------------------------------------------------------------
# 2) Restaurar la base desde el archivo ZIP
# -----------------------------------------------------------------------------------

echo "[init-db] Restaurando la base de datos \"$POSTGRES_DB\" con el dump inicial..."

DUMP_ZIP="/docker-entrypoint-initdb.d/dump.sql.zip"
if [[ ! -f "$DUMP_ZIP" ]]; then
  echo "[init-db] Archivo $DUMP_ZIP no encontrado; omitiendo restauración."
  exit 0
fi

# Descomprimir directo a stdout y alimentar psql con el rol correcto
unzip -p "$DUMP_ZIP" | psql -v ON_ERROR_STOP=1 --username "$DUMP_ROLE" --dbname "$POSTGRES_DB"

echo "[init-db] Restauración completada."
