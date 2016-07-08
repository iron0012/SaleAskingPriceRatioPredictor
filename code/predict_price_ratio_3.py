import cPickle as pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from datetime import date
import re
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import gzip

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
                            current_listing_price,
                            vectorizer):
    '''Prepare a pandas dataframe with the input information with the right format for inserting into
    a trained Random Forest model to predict the sale_price to askling_price ratio.
    input:
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

    '''
    df_zip = pd.DataFrame(index = [1], columns = zip_codes)
    df_zip.fillna(0, inplace = True)

    X_stem_lem = vectorizer.transform([input_text])
    features = vectorizer.get_feature_names()
    df_text = pd.DataFrame(X_stem_lem.toarray(), index = [1], columns = features)
    df_text.drop(additional_stop_words, axis = 1, inplace = True)

    df_nontxt = pd.DataFrame(index = [1], columns = non_text_features)
    df_nontxt.set_value(1, non_text_features, [home_size_sq_ft,
                                               number_of_bedrooms,
                                               number_of_bathrooms,
                                               current_listing_price]
                       )
    df_nontxt

    df = pd.concat([df_zip, df_text, df_nontxt],
                  axis = 1)

    return df

def create_feature_importance_dict(df, model):
    '''Accepts a dataframe which is used to extract the feature names (tokenized text features) and a model
    which is used to extract the feature_importances.  Combines the two to form a list of feature importances
    with feature names ranked by the feature importance.  Then uses this list to generate a dictionary of the feature'''
    feature_importances_dict = {}

    #This creates a preliminary list of [rank of feature_imp, (feature_name, feature_imp_value)] that is sorted by rank.
    feature_importances_prelim = zip(range(df.shape[1]),
                                     sorted(zip(df.columns, model.feature_importances_),
                                            key = lambda x: x[1],
                                            reverse = True
                                           )
                                    )

    #This flattens the preliminary list to create a list of [feature_name, rank of feature_imp, feature_imp_value].
    feature_importances_list = [(str(x[1][0]), x[0] + 1, x[1][1]) for x in feature_importances_prelim]

    #From this list, create a dictionary in which the key is the feature_name and the values are
    #rank of feature_importance and the value of feature_importance
    for feature in feature_importances_list:
        feature_importances_dict[feature[0]] = feature[1], feature[2]

    return  feature_importances_dict, feature_importances_list

def feature_occurrence(feature_name, df_feature_occurrence):
    '''This function accepts a feature name and returns the occurrences for in the training data'''
    return df_feature_occurrence[feature_name]


def find_mean_ratio_feature(feature, df):
    return df[df[feature] > 0]['price_ratio'].mean()


def features_present(vectorizer, input_text):
    '''Accepts a trained vectorizer and a text to be transformed into an array by the vectorizer.
    Returns a list of tuples of tokenized text features and each of their occurrences in the text arranged in
    descending order by occurrences.'''
    return sorted(zip(vectorizer.get_feature_names(), vectorizer.transform([input_text]).toarray().tolist()[0]),
                  key = lambda x: x[1],
                  reverse = True
                 )


def Create_text_feat_imp_Dataframe(features_present_list):
    '''Create Dataframe of Text_features_present, Feature_importances, average sale-price/asking-price ratio for a
    particular text_feature, occurrences.
    Input:
    features_present_list: A list of tokenized texts that are present in the input text.

    Output:
    df_feature_present: a Dataframe of Text_features_present, Feature_importances, average sale-price/asking-price ratio for a
    particular text_feature, occurrences.
    '''

    features_dict_list = []
    #Names of the columns in dataframe.
    cols = ['Text feature',
            'Feature importance (10^-3)',
            'Sale-price/asking-price ratio (Avg. = 1.065)',
            'Occurrences in the training data (13,335 properties)']

    for feature in features_present_list:
        #Create a dictionary of values for a given tokenized text feature
        dict_temp = {cols[0]: feature.upper(),
                     cols[1]: feature_importances_dict[feature][1] * 1000,
                     cols[2]: str(find_mean_ratio_feature(feature, df_X_regr_all))[0:8],
                     cols[3]:  str(feature_occurrence(feature, df_feature_occurrence))
                    }
        #Append the dictionary to a list which will be used to create the dataframe
        features_dict_list.append(dict_temp)

    #Create dataframe, sort order by Feature_importance
    df_feature_present = pd.DataFrame(features_dict_list, columns = cols )
    df_feature_present.set_index('Text feature',   inplace = True)
    df_feature_present.sort_values('Feature importance (10^-3)',
                                    axis = 0,
                                   inplace = True,
                                   ascending = False
                                  )
    return df_feature_present

'''Load the list of all zip codes used in training the model'''
filepath = 'list_of_zip_codes'
zip_codes = pickle.load(open(filepath, 'rb'))

'''Load the trained RandomForestRegressor model'''
filepath = 'rf_zip_full_dict_400_50_7.gzip'
rf_zip_full_dict_400_50_7 = pickle.load(gzip.open(filepath, 'rb'))

'''Query the user on the number_of_bedrooms, number_of_bathrooms, zip_code,
home_size_sq_ft, and current_listing_price of the property.'''
number_of_bedrooms = input('Input the number of bedrooms of the property: ')
number_of_bathrooms = input('Input the number of bathrooms of the property: ')
zip_code = input('Input the 5-digit zip code of the property: ')
zip_code = str(zip_code)
home_size_sq_ft = input('Input the size of the home in square feet (interior): ')
current_listing_price = input('Input the listing price: ')

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
non_text_features = ['number_of_bedrooms', 'number_of_bathrooms', 'home_size', 'current_listing_price']

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
                            current_listing_price,
                            vectorizer)


'''Pass DataFrame (converted to a numpy array) to the RF model to predict the
price_ratio'''
predicted_price_ratio = rf_zip_full_dict_400_50_7.predict(df.values)
sale_price = predicted_price_ratio[0] * current_listing_price

'''Output prediction to user'''
print
print
print ('Our model estimates the sale price will be $' + str(int(sale_price)) + \
       ', which is ' + str(predicted_price_ratio[0])[0:6] + ' times the asking price.')

'''Create a list of tokenized_features that are present in the input_text'''
features_present_list = features_present(vectorizer, input_text)
features_present_list = [str(feature[0]) for feature in features_present_list if feature[1] > 0]
features_present_list = [feature for feature in features_present_list if feature not in additional_stop_words]

filepath = 'df_sum.p'
f = open(filepath, 'rb')
df_feature_occurrence = pickle.load(f)
f.close()

df_X_regr_all = pd.read_csv('df_X_regr_all.csv')

feature_importances_dict, feature_importances_list = create_feature_importance_dict(df, rf_zip_full_dict_400_50_7)

print 'Below are a table of expressions tokenized from your input text.'
print 'Note that the "importance" of an expression is a combination of how'
print 'far the price_ratio with the expression is different from the average'
print 'price_ratio for all data (1.065) and how often the expression occurs '
print 'in all data (13,335).'
print Create_text_feat_imp_Dataframe(features_present_list)
# for feature in features_present_list:
#     print feature.upper(),
#     print 'has an importance of ',
#     print str(feature_importances_dict[feature][1]) + '.'
#     print 'The average sale/asking price ratio for properties with this feature is ',
#     print str(find_mean_ratio_feature(feature, df_X_regr_all))[0:8] + '.'
#     print 'It occurred in' + str(feature_occurrence(feature, df_feature_occurrence)) + ' rows in 13,335 rows of the training data.'
#     print
#     print
