#!/bin/bash
#set -e
exec python manage.py db upgrade &
exec python manage.py db init &
exec python manage.py db upgrade &
exec python manage.py db migrate &
exec python add_new_API_table.py &
exec python app.py
#exec "$@"
