import re
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import pandas as pd

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
