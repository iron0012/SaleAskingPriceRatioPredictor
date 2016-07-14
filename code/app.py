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
import cPickle as pickle
app = Flask(__name__)#,  static_url_path = "/./static", static_folder = "static")


#load trained random forest
filepath = 'rf_zip_full_dict_400_50_7.gzip'
rf_zip_full_dict_400_50_7 = pickle.load(gzip.open(filepath, 'rb'))

filepath = 'df_sum.p'
f = open(filepath, 'rb')
df_feature_occurrence = pickle.load(f)
f.close()


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
filepath = 'df_X_regr_all.gzip'
f = gzip.open(filepath, 'rb')
df_X_regr_all = pickle.load(f)
f.close()

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
        features_present_list = features_present(vectorizer, form.text.data)
        features_present_list = [str(feature[0]) for feature in features_present_list if feature[1] > 0]
        features_present_list = [feature for feature in features_present_list if feature not in additional_stop_words]
        feature_importances_dict, feature_importances_list = create_feature_importance_dict(df, rf_zip_full_dict_400_50_7)
        feature_dataframe = Create_text_feat_imp_Dataframe(features_present_list, feature_importances_dict, df_X_regr_all, df_feature_occurrence)
        sale_price = str(int(sale_price))
        price_ratio =str(price_ratio)[0:5]
        return render_template('view.html', form=form, price_ratio=price_ratio, sale_price=sale_price, features=feature_dataframe.to_html(), scroll="result")
    else:
        price_ratio = None
        sale_price = None
        feature_dataframe = None
        return render_template('view.html', form=form, price_ratio=price_ratio, sale_price=sale_price, features=None)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8037, debug=True)
