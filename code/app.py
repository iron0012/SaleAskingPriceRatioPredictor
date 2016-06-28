from collections import Counter
from flask import Flask, request
app = Flask(__name__)


def dict_to_html(d):
    return '<br>'.join('{0}: {1}'.format(k, d[k]) for k in sorted(d))


# Form page to submit text
@app.route('/')
def submission_page():
    return '''
        <form action="/predictor" method='POST' >
            Text <input type="text" name="text" /> </br>
            Rooms number <input type="text" name ="rooms" /> </br>
            Bathrooms number <input type="text" name ="bathrooms" /> </br>
            ZIP <input type="text" name ="zip" /> </br>
            Sqf <input type="text" name ="sqf" /> </br>
            <input type="submit" value="Submit" />
        </form>
        '''


# My word counter app
@app.route('/predictor', methods=['POST'])
def predictor():
    text = str(request.form['text'])
    rooms = str(request.form['rooms'])
    bathrooms = str(request.form['bathrooms'])
    ZIP = str(request.form['zip'])
    sqf = str(request.form['sqf'])

    # TODO: Implement here the prediction: 1 - upload your models
    #                                      2 - apply them on the input data (remember to change the type of the data i.e from
    #                                               string to float etc...)
    return "The predicted value is", prediction


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
