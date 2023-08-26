from app import app
from forms import ImageSearchForm
from flask import flash, render_template, request, redirect
import flask
import os
import clipthing as c

clip = c.ClipThing()
clip.read_dir()


@app.route('/', methods=['GET', 'POST'])
def index():
    search = ImageSearchForm(request.form)
    if request.method == 'POST':
        return search_results(search, request.form)
    return render_template('index.html', form=search)


@app.route('/results')
def search_results(search, a):
    folderpath = os.path.dirname(os.path.realpath(__file__))
    results = []
    search_string = search.data['search']

    if search.data['search'] == '':
        #flash('NO SEARCH :(')
        return redirect('/')

    if not results:
        global clip
        _,out_js,IDs = clip.searchtext(search_string,50)   
        path = folderpath + "\\static\\image.json"
        clip.writejson(out_js,path)
        path = folderpath + "\\static\\smolimage.json"
        clip.writejson(IDs,path)
        #flash("FOUND STUFF!")
        return redirect('/')
    else:
        return redirect('/')

@app.route('/more')
def more():
    folderpath = os.path.dirname(os.path.realpath(__file__))
    ID = flask.request.args.get('ID')
    sim = flask.request.args.get('sim')
    print(ID)
    print(type(ID))
    _,out_js,IDs = clip.searchID(ID,50)
    path = folderpath + "\\static\\image.json"
    clip.writejson(out_js,path)
    path = folderpath + "\\static\\smolimage.json"
    clip.writejson(IDs,path)
    # flash("PRESS SEARCH!")
    # index()
    return ("ID SEARCH DONE!")


@app.route('/compare')
def compare():
    raise ValueError

if __name__ == '__main__':
    import os
    app.run()