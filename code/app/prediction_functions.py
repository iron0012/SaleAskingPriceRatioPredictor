import re
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import pandas as pd
from wtforms import Form, FloatField, validators, StringField, IntegerField
import cPickle as pickle


WORD_PATTERN = re.compile("(^|\s+)([A-Za-z]+)")
STEMMER = SnowballStemmer("english")
LEMMER = WordNetLemmatizer()

#Combine two sets of stop words
STOPWORDS = stopwords.words('english')
filepath_zip = 'list_of_zip_codes'

zip_codes = pickle.load(open(filepath_zip, 'rb'))







class InputForm(Form):
    text = StringField(
        label='text', default="",
        validators=[validators.InputRequired()])
    number_of_bedrooms = IntegerField(
        label='Number of bedrooms',
        validators=[validators.InputRequired()])
    number_of_bathrooms = IntegerField(
        label='Number of bathrooms',
        validators=[validators.InputRequired()])
    home_size_sq_ft = FloatField(
        label = 'Size of the home in square feet',
        validators=[validators.InputRequired()])
    zip_code = StringField(
        label='Zip code', default="",
        validators=[validators.InputRequired()])
    current_listing_price = FloatField(
        label='Current listing price',
        validators = [validators.AnyOf(zip_codes)])




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
