from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__, template_folder='templates')

model = pickle.load(open('CTindex.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    if request.method == 'POST':
        aggregate = request.form['Aggregate']
        source = request.form['source']
        viscosity = float(request.form['viscosity'])  # Assuming viscosity is a float
        dag = request.form['DAG']
        air_voids = request.form['air_voids']
        
        # Convert input data to a format that the model expects
        input_data = np.array([[aggregate, source, viscosity, dag, air_voids]])
        
        # Make prediction using the model
        pred = model.predict(input_data)
        
        processed_data = pred  # Example processing
        
        return render_template('result.html', input_data=input_data, processed_data=processed_data)
    # return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
