from flask import Flask, render_template,request
from args import status_mapping,location_mapping,property_type_mapping,direction_mapping
import pickle
import numpy as np

with open('Model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('Scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
    
app = Flask(__name__)

@app.route('/',methods=['GET','POST'])
def index():
    print(request.method)
    print(request.form)
    if request.method == 'POST':
        
        bedroom = request.form['bedrooms']
        bathroom = request.form['bathrooms']
        loc = request.form['location']
        sqft = request.form['sft']
        status = request.form['status']
        dir = request.form['direction']
        prop = request.form['property type']

        input_array = np.array([[
            bedroom,
            bathroom,
            loc,
            sqft,
            status,
            dir,
            prop         
        ]])

        input_df = scaler.transform(input_array)
        prediction = model.predict(input_df)[0]
        
        return render_template(
            'index.html',
            location_mapping = location_mapping,
            property_type_mapping = property_type_mapping,
            status_mapping = status_mapping,
            direction_mapping = direction_mapping,
            prediction = prediction
            )

    else:
        return render_template(
            'index.html',
            location_mapping = location_mapping,
            property_type_mapping = property_type_mapping,
            status_mapping = status_mapping,
            direction_mapping = direction_mapping
            )

# routes are known as view functions
# routes and function can be named different but it is not advisable

@app.route('/nitin',methods=['GET','POST'])
def nitin():
    return "I am Venkata Rama Nitin Pathuri a final year Data Science student in KKR & KSR Institute of Technology and Sciences <br><a href='./'>Home page</a>"

@app.route('/sravs')
def sravs():
    return "I am Sravan and my classmates calls me Sravs <br><a href='./'>Home page</a>"

@app.route('/khaleel')
def khaleel():
    return "I am Khaleel and my classmates calls me K <br><a href='./'>Home page</a>"

@app.route('/anil')
def anil():
    return "I am Anil and my classmates calls me Anila <br><a href='./'>Home page</a>"

@app.errorhandler(404)
def page_not_found(error):
    return render_template('/404.html'), 404

@app.errorhandler(405)
def page_nfound(error):
    return render_template('/404.html'), 405

@app.errorhandler(500)
def page_notfound(error):
    return render_template('/404.html'), 500

if __name__=='__main__':
    #app.run(use_reloader=True,debug=True)
    app.run()