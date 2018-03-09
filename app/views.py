# -*- coding: utf-8 -*-
import os, json, codecs, datetime, sqlite3
from flask import Flask, request, session, g, redirect, url_for, abort, render_template, flash, jsonify, make_response
import hashlib
from werkzeug.utils import secure_filename
from app import app
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from . import rbfnn

def connect_excel(excel=''):
    if excel == '' or os.path.isfile(excel) != True:
        excel = app.config['DATAEXCEL']
    data = pd.read_csv(excel,';', thousands='.', decimal=',')
    # datakita = data.to_records()
    return data


def connect_db():
    """Connects to the specific database."""
    rv = sqlite3.connect(app.config['DATABASE'])
    rv.row_factory = sqlite3.Row
    return rv

def get_db():
    """Opens a new database connection if there is none yet for the
    current application context.
    """
    if not hasattr(g, 'sqlite_db'):
        g.sqlite_db = connect_db()
    return g.sqlite_db

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def safe_div(x,y):
    if y == 0:
        return 0
    else:
        return x/y

def auth_check(level):
    granted = 0
    if session['logged_in']:
        for lv in level:
            if lv == session['level']:
                granted = granted + 1
    if granted > 0:
        return True
    else:
        return False


@app.teardown_appcontext
def close_db(error):
    """Closes the database again at the end of the request."""
    if hasattr(g, 'sqlite_db'):
        g.sqlite_db.close()


@app.route('/')
def index():
    if session == None:
        return redirect(url_for('login'))
    else:
        return render_template('index.html')

@app.route('/about')
def about():
    db = get_db()
    cur = db.execute('select * from user')
    users = cur.fetchall()
    return render_template("about.html", users=users)

@app.route('/kelolauser')
def kelolauser():
    if auth_check([1]) != True:
        return redirect(url_for('login'))
    db = get_db()
    cur = db.execute('select * from user')
    users = cur.fetchall()
    return render_template("kelolauser.html", users=users)

@app.route('/tambahuser', methods=['GET','POST'])
def tambahuser():
    if auth_check([1]) != True:
        return redirect(url_for('login'))
    if request.method == 'POST':
        db = get_db()
        username = request.form['username']
        passwd = hashlib.md5(request.form['password']).hexdigest()
        jabat = request.form['level']
        db.execute('insert into user (username, password,jabatan) values (?,?,?)', [username, passwd, jabat])
        db.commit()
        flash("Pengguna Baru Berhasil Dibuat!")
        return redirect(url_for('kelolauser'))
    else:
        preform = {'id':'','username':'','password':'','jabatan':'1'}
        return render_template("form_user.html", isi=preform, judul="Tambah Pengguna", valsubmit="Tambah", actsubmit="tambahuser")

@app.route('/edituser', methods=['GET','POST'])
def edituser():
    if auth_check([1]) != True:
        return redirect(url_for('login'))
    if request.method == 'POST':
        db = get_db()
        iduser = request.form['id']
        username = request.form['username']
        jabat = request.form['level']
        db.execute('update user set username = "?" and jabatan = "?" where id = ?', [username, jabat, iduser])
        db.commit()
        flash("Data Pengguna Berhasil Dibuat!")
        return redirect(url_for('kelolauser'))
    else:
        db = get_db()
        cur = db.execute('select * from user where id = ?',request.args['id'])
        user = cur.fetchall()
        return render_template('form_user.html', isi=user[0], judul="Ubah Data Pengguna", valsubmit="Simpan", actsubmit="edituser")


@app.route('/hapususer')
def hapususer():
    if auth_check([1]) != True:
        return redirect(url_for('login'))
    if request.args['id'] != '':
        db = get_db()
        db.execute('delete from user where id = ?',request.args['id'])
        db.commit()
        flash("Pengguna telah dihapus!")
        return redirect(url_for('kelolauser'))

@app.route('/datalatih')
def datalatih():
    namafile = os.path.join(app.config['UPLOAD_FOLDER'],'data-latih.csv')
    data = connect_excel(namafile)
    mixmax = MinMaxScaler()
    datakita = mixmax.fit_transform(pd.DataFrame(data={'production':data['PROUCTION'],'panen':data['PANEN']}, columns=['production','panen']))
    jenis = 'latih'
    return render_template("pilihcsv.html", data=datakita, dataraw=data, jenisdata=jenis, file_ada=os.path.isfile(namafile))

@app.route('/datauji')
def datauji():
    namafile = os.path.join(app.config['UPLOAD_FOLDER'],'data-uji.csv')
    data = connect_excel(namafile)
    mixmax = MinMaxScaler()
    datakita = mixmax.fit_transform(pd.DataFrame(data={'production':data['PROUCTION'],'panen':data['PANEN']}, columns=['production','panen']))
    jenis = 'uji'
    return render_template("pilihcsv.html", data=datakita, dataraw=data, jenisdata=jenis, file_ada=os.path.isfile(namafile))

@app.route('/laporan')
def laporan():
		if auth_check([1,2]) != True:
			return redirect(url_for('login'))
		namafile = os.path.join(app.config['UPLOAD_FOLDER'],'data-uji.csv')
		data = connect_excel(namafile)
		mixmax = MinMaxScaler()
		datakita = mixmax.fit_transform(pd.DataFrame(data={'production':data['PROUCTION'],'panen':data['PANEN']}, columns=['production','panen']))
		file_json = os.path.join(app.root_path,'log-uji.json')
		json_raw = codecs.open(file_json,'r',encoding='utf-8').read()
		param = json.loads(json_raw)
		return render_template('laporan.html', dataparam=param, hitungan=range(len(datakita)))

@app.route('/downloadlap')
def downloadlap():
    if auth_check([1,2]) != True:
        return redirect(url_for('login'))
    file_json = os.path.join(app.root_path,'log-uji.json')
    json_raw = codecs.open(file_json,'r',encoding='utf-8').read()
    param = json.loads(json_raw)
    datareturn = pd.DataFrame(data={'tanggal': param['tanggal'], 'hasil produksi': param['X_raw'], 'hasil panen': param['y_raw'], 'prediksi hasil panen':param['denorm']}, columns=['tanggal','hasil produksi','hasil panen','prediksi hasil panen'])
    response = make_response(datareturn.to_csv(sep=';', decimal=','))
    cd = "attachment; filename=laporan_prediksi.csv"
    response.headers['Content-Disposition'] = cd
    response.mimetype = 'text/csv'
    return response

@app.route('/unggahdata', methods=['POST'])
def unggahdata():
    if request.form['jenisdata'] == 'latih':
        filename = 'data-latih.csv'
        hal = 'datalatih'
    else:
        filename = 'data-uji.csv'
        hal = 'datauji'
    if 'berkas' not in request.files:
        flash('Tidak ada file yang dikirim')
        return redirect(url_for(hal))
    file = request.files['berkas']
    if file.filename == '':
        flash('Tidak ada file terpilih')
        return redirect(url_for(hal))
    if file and allowed_file(file.filename):
        ## filename = secure_filename(file.filename)
        
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        flash('Berhasil mengunggah data')
    else:
        flash('File yang dikirim tidak diperbolehkan')
    return redirect(url_for(hal))

@app.route('/pelatihan', methods=['GET','POST'])
def pelatihan():
    if auth_check([1]) != True:
        return redirect(url_for('login'))
    namafile = os.path.join(app.config['UPLOAD_FOLDER'],'data-latih.csv')
    data = connect_excel(namafile)
    mixmax = MinMaxScaler()
    datakita = mixmax.fit_transform(pd.DataFrame(data={'production':data['PROUCTION'],'panen':data['PANEN']}, columns=['production','panen']))
    file_json = os.path.join(app.root_path,'log.json')
    json_raw = codecs.open(file_json,'r',encoding='utf-8').read()
    param = json.loads(json_raw)
    return render_template('pelatihan.html', dataraw=data, datanorm=datakita, dataparam=param, hitungan=range(len(datakita)))

@app.route('/latihaja',  methods=['POST'])
def latihaja():
    i_node = int(request.form['in_node_param'])
    h_node = int(request.form['hnode_param'])
    max_ep = int(request.form['maxepoch_param'])
    lrate  = float(request.form['lrate_param'])
    namafile = os.path.join(app.config['UPLOAD_FOLDER'],'data-latih.csv')
    data = connect_excel(namafile)
    hasilnya = rbfnn.train(data['PROUCTION'], data['PANEN'], tanggal=data['DATE'], n_hidden=h_node, input_node=i_node, ephocs=max_ep, lr=lrate)
    json.dump(hasilnya, codecs.open('app/log.json', 'w', encoding='utf-8'), separators=(',',':'), sort_keys=True, indent=4)
    #return jsonify(hasilnya)
    return redirect(url_for('pelatihan'));

@app.route('/ujiaja', methods=['GET','POST'])
def ujiaja():
    i_node = int(request.form['in_node_param'])
    h_node = int(request.form['hnode_param'])
    max_ep = int(request.form['maxepoch_param'])
    lrate  = float(request.form['lrate_param'])
    namafile = os.path.join(app.config['UPLOAD_FOLDER'],'data-uji.csv')
    data = connect_excel(namafile)
    file_json = os.path.join(app.root_path,'log.json')
    json_raw = codecs.open(file_json,'r',encoding='utf-8').read()
    param = json.loads(json_raw)
    hasilnya = rbfnn.test(data['PROUCTION'], data['PANEN'], param['weight'], param['center'], tanggal=data['DATE'], n_hidden=h_node, input_node=i_node, ephocs=max_ep, lr=lrate)
    json.dump(hasilnya, codecs.open('app/log-uji.json', 'w', encoding='utf-8'), separators=(',',':'), sort_keys=True, indent=4)
    #return jsonify(hasilnya)
    return redirect(url_for('pengujian'));


@app.route('/pengujian', methods=['GET','POST'])
def pengujian():
    if auth_check([1]) != True:
        return redirect(url_for('login'))
    namafile = os.path.join(app.config['UPLOAD_FOLDER'],'data-uji.csv')
    data = connect_excel(namafile)
    mixmax = MinMaxScaler()
    datakita = mixmax.fit_transform(pd.DataFrame(data={'production':data['PROUCTION'],'panen':data['PANEN']}, columns=['production','panen']))
    file_json = os.path.join(app.root_path,'log-uji.json')
    json_raw = codecs.open(file_json,'r',encoding='utf-8').read()
    param = json.loads(json_raw)
    file_json = os.path.join(app.root_path,'log.json')
    json_raw = codecs.open(file_json,'r',encoding='utf-8').read()
    paramlatih = json.loads(json_raw)
    return render_template('pengujian.html', dataraw=data, datanorm=datakita, dataparam=param, paramlatih=paramlatih, hitungan=range(len(datakita)))

@app.route('/prediksi', methods=['GET','POST'])
def prediksi():
    namafile = os.path.join(app.config['UPLOAD_FOLDER'],'data-latih.csv')
    data = pd.read_csv(namafile,';', thousands='.', decimal=',', parse_dates=[1])
    file_json = os.path.join(app.root_path,'log.json')
    json_raw = codecs.open(file_json,'r',encoding='utf-8').read()
    paramlatih = json.loads(json_raw)
    tgl = data['DATE'].loc[data.shape[0]-1] + datetime.timedelta(1) # default ambil tanggal terakhir di dataset + 1 hari
    if request.method == 'POST':
        tgr = request.form['tanggal'].split('-')
        tgl_temp = datetime.date(int(tgr[0]), int(tgr[1]), int(tgr[2]))
        tgl = tgl_temp if data[data['DATE'] == tgl_temp].shape[0] == 1 else tgl
    tglalu = tgl - datetime.timedelta(days=paramlatih['input_node'])
    mask = (data['DATE'] >= tglalu) & (data['DATE'] < tgl)
    selected_data = data.loc[mask]
    hasil = rbfnn.predict(selected_data['PROUCTION'], selected_data['PANEN'], paramlatih['weight'], paramlatih['center'])
    return render_template('prediksi.html', hasil=hasil, tgl=tgl, paramlatih=paramlatih, sdata=selected_data.reset_index())

@app.route('/login', methods=['GET','POST'])
def login():
    if session == None:
        return redirect(url_for('index'))
    error = None
    if request.method == 'POST':
        db = get_db()
        cur = db.execute('select * from user where username = ?', [request.form['username']])
        fetched = cur.fetchall()
        if len(fetched) != 0:
            user = fetched[0]
            if hashlib.md5(request.form['password']).hexdigest() == user['password']:
                session['logged_in'] = True
                session['level'] = user['jabatan']
                session['usrlog'] = user['username']
                flash("Anda telah berhasil masuk")
                return redirect(url_for('index'))
            else:
                flash("Username/Password yang Anda masukkan salah")
        else:
            flash("Username/Password yang Anda masukkan salah")
    return render_template('login.html', error=error)

@app.route('/logout')
def logout():
    session.pop('logged_in',None)
    session.pop('level',None)
    session.pop('usrlog',None)
    flash("Anda telah berhasil keluar")
    return redirect(url_for('index'))