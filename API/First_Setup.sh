#!/bin/bash
#set -e
exec python manage.py db upgrade &
exec python manage.py db init &
exec python manage.py db upgrade &
exec python manage.py db migrate &
#exec python app.py
#exec "$@"
