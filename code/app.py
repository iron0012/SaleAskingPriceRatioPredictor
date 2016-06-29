from collections import Counter
import cPickle as pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import matplotlib.pyplot as plt
from datetime import date
from sklearn.feature_extraction.text import CountVectorizer
from flask import Flask, request
from prediction_functions import *
app = Flask(__name__)

# zip data
filepath_zip = 'list_of_zip_codes'
zip_codes = pickle.load(open(filepath_zip, 'rb'))
# vectorizer and stemming
filepath_vect = 'agent_desc_vectorizer.p'
vectorizer = pickle.load(open(filepath_vect, 'rb'))

additional_stop_words  =  ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday', 'offer',
                               'account', 'disclosur', 'due', 'date', 'sfar', 'broker tour', 'pm', 'offer review',
                               'tour', 'offer date', 'offer due', 'pre', 'accept', 'call', 'close', 'pleas call',
                               'noon', 'open', 'price', 'zestim', 'zestim accur', 'zestim forecast', u'zestim forecastcr',
                               'zestim home', 'zestim owner', 'zestim rent', 'zestim see', 'zestim valu','zestim zestim',
                               'zestim zillow', 'zillow', 'zillow estim', 'zillow valu'
                               ]
non_text_features = ['number_of_bedrooms', 'number_of_bathrooms', 'home_size']

#load trained random forest
filepath_model = 'rf_zip_txt_nontxt_model.p'
rf_final_model_100_50_7 = pickle.load(open(filepath_model, 'rb'))

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
    # getting input data
    input_text = str(request.form['text'])
    number_of_bedrooms = int(request.form['rooms'])
    number_of_bathrooms = int(request.form['bathrooms'])
    zip_code = str(request.form['zip'])
    home_size_sq_ft = float(request.form['sqf'])

    X_stem_lem = vectorizer.transform([input_text])
    # additional stop words

    #
    df =create_dataframe_webapp(zip_codes,
                                non_text_features,
                                additional_stop_words,
                                input_text,
                                zip_code,
                                home_size_sq_ft,
                                number_of_bedrooms,
                                number_of_bathrooms,
                                vectorizer)


    price_ratio = rf_final_model_100_50_7.predict(df.values)[0]

    return "The predicted price ration value is %s" % price_ratio


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
