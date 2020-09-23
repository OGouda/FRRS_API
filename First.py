import os

try:
    os.system('python manage.py db upgrade')
    os.system('python manage.py db init')
    os.system('python manage.py db migrate')
    os.system('python add_new_API_table.py')
except:
    pass
os.system('python app.py')


