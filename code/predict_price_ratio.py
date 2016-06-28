import cPickle as pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import matplotlib.pyplot as plt
from datetime import date
import re
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer


WORD_PATTERN = re.compile("(^|\s+)([A-Za-z]+)")
STEMMER = SnowballStemmer("english")
LEMMER = WordNetLemmatizer()

#Combine two sets of stop words
STOPWORDS = stopwords.words('english')

#Define tokenizers that do stemming only, lemmatizing only, and both
def tokenize_stem(s):
    return  [STEMMER.stem(match.group(2)) \
             for match in WORD_PATTERN.finditer(s) \
             if match.group(2) not in STOPWORDS and len(match.group(2)) >= 2]

def tokenize_lem(s):
    return [LEMMER.lemmatize(match.group(2)) \
            for match in WORD_PATTERN.finditer(s) \
            if match.group(2) not in STOPWORDS and len(match.group(2)) >= 2]

def tokenize_stem_lem(s):
    return [STEMMER.stem(LEMMER.lemmatize(match.group(2))) \
            for match in WORD_PATTERN.finditer(s) \
            if match.group(2) not in STOPWORDS and len(match.group(2)) >= 2]

def create_dataframe_webapp(zip_codes,
                            non_text_features,
                            additional_stop_words,
                            input_text,
                            zip_code,
                            home_size_sq_ft,
                            number_of_bedrooms,
                            number_of_bathrooms,
                            vectorizer):
    '''Prepare a pandas dataframe with the input information with the right format for inserting into
    a trained Random Forest model to predict the sale_price to askling_price ratio.

    1.  Recreate a dummyfied dataframe of all 71 zip codes in our data called df_zip.
    2.  Pass the input text (agent's comments) into the the vectorizer transform function.
        Combine them with the feature names (from the vectorizer object) and create a dataframe
        called df_text.
    3.  Create a dataframe that contains the user-input info of: a  number_of_bedrooms
                                                                 b  number_of_bathrooms
                                                                 c  home_size_sq_ft
    4.  Concatenate the above three dataframes and return


    Input:
    zip_codes:  a LIST of zip codes that are in random forest model.
    non_text_features: a LIST of text of the names of the non-text features, e.g. 'home_size', 'number_of_bedrooms'.
    additional_stop_words:  a LIST of tokeized_expression that will be removed from the final input matrix.

    (The following pertains specific to the home whose price_ratio will be predicted )
    input_text: The agent's description of the home.
    zip_code:  The 5-digit zip_code of the home (text).
    home_size_sq_ft: home size in square feet (integer).
    number_of_bedrooms: number of bedrooms (float).
    number_of_bathrooms: number of bathrooms (float)

    vectorizer: vectorizer to convert input text of the agent's description to the vector of tekenized expressions.

    Output:
    df: A pandas dataframe that contains all the input data in a format ready to be input into the random forest regressor
        predictor.



    '''
    # Recreate a dummyfied dataframe of all 71 zip codes in our data called df_zip.
    df_zip = pd.DataFrame(index = [1], columns = zip_codes)
    df_zip.fillna(0, inplace = True)

    #Pass the input text (agent's comments) into the the vectorizer transform function.
    #Combine them with the feature names (from the vectorizer object) and create a dataframe called df_text.
    X_stem_lem = vectorizer.transform([input_text])
    features = vectorizer.get_feature_names()
    df_text = pd.DataFrame(X_stem_lem.toarray(), index = [1], columns = features)
    df_text.drop(additional_stop_words, axis = 1, inplace = True)


    # Create a dataframe that contains the user-input info of: a  number_of_bedrooms
    #                                                          b  number_of_bathrooms
    #                                                          c  home_size_sq_ft
    df_nontxt = pd.DataFrame(index = [1], columns = non_text_features)
    df_nontxt.set_value(1, non_text_features, [number_of_bedrooms, number_of_bathrooms, home_size_sq_ft])
    df_nontxt

    #Concatenate the above three dataframes and return
    df = pd.concat([df_zip, df_text, df_nontxt],
                  axis = 1)
    return df


'''Load the list of all zip codes used in training the model'''
filepath = 'list_of_zip_codes'
zip_codes = pickle.load(open(filepath, 'rb'))


'''Query the user on the number_of_bedrooms, number_of_bathrooms, the zip_code,
and the home_size_sq_ft of the property.'''
number_of_bedrooms = input('Input the number of bedrooms of the property: ')
number_of_bathrooms = input('Input the number of bathrooms of the property: ')
zip_code = input('Input the 5-digit zip code of the property: ')
zip_code = str(zip_code)
home_size_sq_ft = input('Input the size of the home in square feet (interior): ')


'''Ask the user to input the agent's description of the property.'''
input_text = raw_input("Input the text of the agent's description (no quotes): ")


'''Import the word vectorizer.  Use it to vectorize the agent's comments.'''
filepath = 'agent_desc_vectorizer.p'
vectorizer = pickle.load(open(filepath, 'rb'))
X_stem_lem = vectorizer.transform([input_text])

'''These are additional words that we found would confuse the model and are to
be removed from the dataframe'''
additional_stop_words  =  ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday', 'offer',
                           'account', 'disclosur', 'due', 'date', 'sfar', 'broker tour', 'pm', 'offer review',
                           'tour', 'offer date', 'offer due', 'pre', 'accept', 'call', 'close', 'pleas call',
                           'noon', 'open', 'price', 'zestim', 'zestim accur', 'zestim forecast', u'zestim forecastcr',
                           'zestim home', 'zestim owner', 'zestim rent', 'zestim see', 'zestim valu','zestim zestim',
                           'zestim zillow', 'zillow', 'zillow estim', 'zillow valu'
                           ]
non_text_features = ['number_of_bedrooms', 'number_of_bathrooms', 'home_size']

'''Prepare a pandas dataframe with the input information with the right format for inserting into
a trained Random Forest model to predict the sale_price to askling_price ratio.'''
df =create_dataframe_webapp(zip_codes,
                            non_text_features,
                            additional_stop_words,
                            input_text,
                            zip_code,
                            home_size_sq_ft,
                            number_of_bedrooms,
                            number_of_bathrooms,
                            vectorizer)


'''Load the trained RandomForestRegressor model'''
filepath = 'rf_zip_txt_nontxt_model.p'
rf_final_model_100_50_7 = pickle.load(open(filepath, 'rb'))


'''Pass DataFrame (converted to a numpy array) to the RF model to predict the
price_ratio'''
price_ratio = rf_final_model_100_50_7.predict(df.values)

'''Output prediction to user'''
print
print
print ('This model estimates the sale price will be ' + str(price_ratio[0])[0:6] + ' times the asking price.')
