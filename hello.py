from flask import Flask, redirect, request, flash
from catboost import CatBoostClassifier
import os
from werkzeug.utils import secure_filename
import pandas as pd
from sqlalchemy import create_engine


app = Flask(__name__)
# SECRET_KEY = os.urandom(32)
# app.config['SECRET_KEY'] = SECRET_KEY
UPLOAD_FOLDER = ''
ALLOWED_EXTENSIONS = {'csv', 'txt'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def data_preproccesing(df):
    df.loc[df.TotalSpent.astype(str).str.len() == 1, 'TotalSpent'] = \
        df.loc[df.TotalSpent.astype(str).str.len() == 1, 'MonthlySpending'].map(lambda x: x)
    df.TotalSpent = df.TotalSpent.astype('float')
    return df


def get_db_connection():
    return create_engine('postgresql://ivanovd:ivanovd@db/postgres')


model = CatBoostClassifier()
model.load_model('/app/model')


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
            df = pd.read_csv(file.filename)
            df = data_preproccesing(df)
            preds = model.predict_proba(df)[:, 1]
            df['Predicts'] = preds
            df.to_sql('Test', con=get_db_connection(), if_exists='append')
            return 'file uploaded and processed'
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''


if __name__ == '__main__':
    app.run(host='0.0.0.0', port='5000')