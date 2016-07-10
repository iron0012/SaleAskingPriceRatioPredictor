from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import matplotlib.pyplot as plt
from datetime import date
from sklearn.feature_extraction.text import CountVectorizer
from flask import Flask, request, render_template
import gzip
from prediction_functions import *
app = Flask(__name__,  static_url_path = "/./static", static_folder = "static")

#load trained random forest
filepath = 'rf_zip_full_dict_400_50_7.gzip'
rf_zip_full_dict_400_50_7 = pickle.load(gzip.open(filepath, 'rb'))

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
non_text_features = ['number_of_bedrooms', 'number_of_bathrooms', 'home_size', "current_listing_price"]


# Form page to submit text
# This part is an html of the submission_page
@app.route('/', methods=['GET', 'POST'])
def index():
    form = InputForm(request.form)
    if request.method == 'POST': #and form.validate():
        df =create_dataframe_webapp(zip_codes,
                                        non_text_features,
                                        additional_stop_words,
                                        form.text.data,
                                        form.zip_code.data,
                                        form.home_size_sq_ft.data,
                                        form.number_of_bedrooms.data,
                                        form.number_of_bathrooms.data,
                                        form.current_listing_price.data,
                                        vectorizer)

        price_ratio = rf_zip_full_dict_400_50_7.predict(df.values)[0]
        sale_price =  price_ratio * form.current_listing_price.data
    else:
        price_ratio = None
        sale_price = None

    return render_template('view.html', form=form, price_ratio=price_ratio, sale_price=sale_price)


#
# def submission_page():
#     return '''
#         <form action="/predictor" method='POST' >
#             Text <input type="text" name="text" /> </br>
#             Rooms number <input type="text" name ="rooms" /> </br>
#             Bathrooms number <input type="text" name ="bathrooms" /> </br>
#             ZIP <input type="text" name ="zip" /> </br>
#             Sqf <input type="text" name ="sqf" /> </br>
#             Listing price <input type="text" name="input_price" /> </br>
#             <input type="submit" value="Submit" />
#         </form>
#         '''


# My word counter app
# @app.route('/predictor', methods=['POST'])
# def predictor():
#     # getting input data
#     input_text = str(request.form['text'])
#     number_of_bedrooms = int(request.form['rooms'])
#     number_of_bathrooms = int(request.form['bathrooms'])
#     zip_code = str(request.form['zip'])
#     home_size_sq_ft = float(request.form['sqf'])
#     current_listing_price = float(request.form['input_price'])
#     X_stem_lem = vectorizer.transform([input_text])
#     # additional stop words
#
#     #
#     df =create_dataframe_webapp(zip_codes,
#                                 non_text_features,
#                                 additional_stop_words,
#                                 input_text,
#                                 zip_code,
#                                 home_size_sq_ft,
#                                 number_of_bedrooms,
#                                 number_of_bathrooms,
#                                 current_listing_price,
#                                 vectorizer)
#
#
#     price_ratio = rf_zip_full_dict_400_50_7.predict(df.values)[0]
#     sale_price =  price_ratio * current_listing_price
#
#     return "Our model estimates the sale price will be  %s, which is %s times the asking price." % (sale_price , price_ratio)
#

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
