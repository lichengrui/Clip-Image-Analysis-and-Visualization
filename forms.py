# forms.py

from wtforms import Form, StringField, SelectField, validators

class ImageSearchForm(Form):
    select = SelectField('Search:')
    search = StringField('')