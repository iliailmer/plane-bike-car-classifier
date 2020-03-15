import os
from flask import Flask, flash, request, redirect, render_template
from werkzeug.utils import secure_filename
from werkzeug import SharedDataMiddleware
import torch
from skimage.io import imread
from fastai.vision import *
from fastai import *
from fastai.callbacks.hooks import *

device = torch.device('cpu')

UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

app.add_url_rule('/uploads/<filename>', 'uploaded_file',
                 build_only=True)
app.wsgi_app = SharedDataMiddleware(app.wsgi_app, {
    '/uploads': app.config['UPLOAD_FOLDER']
})


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            path = './uploads'
            _data = open_image(f'{path}/{file.filename}')
            _learner = load_learner(Path('./model/'))
            _learner.data.classes = ['bike', 'car', 'plane']
            cls, _, outputs = _learner.predict(_data)
            pred_list = sorted(
                zip(_learner.data.classes, map(float, outputs)),
                key=lambda p: p[1],
                reverse=True
            )
            result_dict = {a: '{:.8f}%'.format(b*100)
                           for a, b in pred_list}  # {
            # "Predictions":}

            return render_template("output.html", filename=filename,
                                   output=result_dict)
            # jsonify(result_dict)  # shamelessly stolen the output from https://github.com/simonw/cougar-or-not

    return render_template("index.html")


if __name__ == '__main__':
    # check if uploads is non-empty and delete everything
    files = os.listdir(Path('./uploads'))
    for each in files:
        os.remove(Path('./uploads') / each)
    app.run(debug=True)
