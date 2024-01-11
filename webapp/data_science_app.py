from flask import Flask, render_template, request
import simple_linear_srv, multiple_linear_srv, naive_bayes_srv, tensorflow_srv

# All the ML services are integrated into this app.
app = Flask(__name__)

@app.route('/')

# The five REST APIs below return the HTML format results 
@app.route('/index')
def index_func():
    return render_template('index.html', title='Home')

@app.route('/simple')
def simple_req():
    return render_template('simple_linear_req.html', title='Simple')

@app.route('/multiple')
def multiple_req():
    return render_template('multiple_linear_req.html')

@app.route('/naive_bayes')
def naive_bayes_req():
    return render_template('naive_bayes_req.html')

@app.route('/tf_iris')
def tf_iris_req():
    return render_template('tf_iris_req.html')

# The five REST APIs below return the JSON format results 
@app.route('/linear/simple', methods=['POST'])
def simple():
    return simple_linear_srv.simple_linear(request)

@app.route('/linear/multiple', methods=['POST'])
def multiple():
    return multiple_linear_srv.multiple_linear(request)

@app.route('/bayes', methods=['POST'])
def bayes():
    return naive_bayes_srv.bayes(request)

@app.route('/tf_iris', methods=['POST'])
def predict():
    return tensorflow_srv.predict(request)

if __name__ == '__main__':
    app.run(debug=True, port=5010)
