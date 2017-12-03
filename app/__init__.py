import os
from flask import Flask
app = Flask(__name__, instance_relative_config=True)

from app import views,model

app.config.from_object('config')
app.config.update(dict(
    DATABASE=os.path.join(app.root_path, 'kelapa.db'),
    DATAEXCEL=os.path.join(app.root_path, 'data-sample.csv'),
    UPLOAD_FOLDER = os.path.join(app.root_path, 'data/')
))
