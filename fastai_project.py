import os
from flask import Flask, flash, request, redirect, render_template, jsonify
from werkzeug.utils import secure_filename
from werkzeug import SharedDataMiddleware
from fastai.vision import open_image, load_learner, Path

UPLOAD_FOLDER = 'fastai_project/uploads'
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
            path = Path('./fastai_project/uploads')
            _data = open_image(path / file.filename)
            _learner = load_learner(Path('./fastai_project/model/'))
            _, _, outputs = _learner.predict(_data)
            _learner.data.classes = ['bicycle', 'car', 'plane']
            os.remove(path / file.filename)
            return jsonify({
                "predictions": sorted(
                    zip(_learner.data.classes, map(float, outputs)),
                    key=lambda p: p[1],
                    reverse=True
                )}
            )  # shamelessly stolen the output from https://github.com/simonw/cougar-or-not

    return render_template("home.html")


if __name__ == '__main__':
    app.run(debug=True)
