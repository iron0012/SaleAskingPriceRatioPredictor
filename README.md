# Sale price vs. asking price of real estate in San Francisco, San Mateo, and Marin counties.
Overview
========
Background
----------

- In many markets, sales prices are generally different from asking prices.
- This sale price/asking price ratio (hereafter referred to as *price ratio*) is inherent difficult to predict because it depends not only on economic demand, but also the individual strategy of the agent or the seller.  Our goal is to try to incorporate this aspect of the process into the model.
- It is expected that the final sale price of a property depends on many factors, so we cannot expect the agent's description to have very high predictive power, especially since all of them are prone to embellish the attractions and diminish the flaws of the property.  Nevertheless, we do expect the description to shed light on the condition of the property.  Here we attempt to answer the question: *How much does an agent’s description of a property have any predictive value on price?*
- Objectives:
	1.  To use Natural Language Processing and Machine Learning to find out if there is the agents' description of a property has any predictive power on how much the *sale* price will go above or below asking (i.e., the sale price/asking price ratio)
	2.  To construct a webapp that allows the user to input relevant information and outputs the predicted sale price.

Executive summary
-----------------
1. With the tokenized text features only, the accuracy for a hold-out set of data for our trained model is 0.103.
2. The accuracy improves to 0.134 if the *basic features* of the property are included in the model.  (Basic features are : zip code, home size (sq. ft.), number of bedrooms, number of bathrooms, asking price.)
3. The accuracy of a model with only the *basic features* is 0.124.

This means the tokenized agents’ descriptions by themselves can explain 10.3% of the variance in the predictions.

Analysis
========
Data
----
- 13,335 properties from 2014-16 in the the counties of San Mateo, San Francisco, and Marin.
- Single family residences only. (Condos and townhouses are excluded, although some slipped through the cracks.)
- Zip code, home size, number of bedrooms, number of bathrooms, and asking prices are included in the training data.
- The average price ratio of of all properties is 1.065.  However there is a large concentration at price ratio = 1.
![Distibution of price ratio](images/price_ratio_distribution.png>)

Vectorization of the agents' descriptions of the property.
----
- Employed the Countvectorizer of the scikitlearn library.  (We experimented with TfidfVectorizer but found no improvement)
- Included 1- and 2-grams only.
- Maximum document frequency (max_df) = 0.9, min_df=75.
- STEMMER = nltk's SnowballStemmer("english")
- LEMMER = nltk's WordNetLemmatizer()
- stopwords = nltk's stopwords.words('english')
- Manually removed the following obvious irrelevant tokenized features:
'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday',
'offer', 'account', 'disclosur', 'due', 'date', 'sfar', 'broker tour', 'pm',
'offer review', 'tour', 'offer date', 'offer due', 'pre', 'accept', 'call',
'close', 'pleas call', 'noon', 'open', 'price', 'yr', 'zestim', 'zestim accur',
'zestim forecast', 'zestim forecastcr', 'zestim home', 'zestim owner',
'zestim rent', 'zestim see', 'zestim valu', 'zestim zestim', 'zestim zillow',
'zillow', 'zillow estim', 'zillow valu'.

Modeling
--------
- We experimented with scikitlearn's RandomForestRegressor and GradientBoostingRegressor at the beginning, but decided to focused on only the RandomForestRegressor because the results between the two are of little difference.

- We included other basic property features such as the zip code, number of bedrooms, number of bathrooms, home size (sq. ft.), and asking price because we know they have predictive powers that will likely reinforce the tokenized text features.  For example, we can observe that the price ratio of a property goes down as the asking price goes up.
![Price ratio vs. asking price](images/Price_ratio_vs_asking_price.png>)
![Price ratio vs. number of bedrooms](images/Price_ratio_vs_bedrooms.png>)
![Price ratio vs. number of bathrooms](images/Price_ratio_vs_bathrooms.png>)

- In general the accuracy does not improve beyond 400 trees (number of estimators). We fixed the the minimum number of samples in newly created leaves to be the square root of maxiumum feature size.  

- The accuracies of random forest models at different max feature size/min leaves with text features only, basic features only and both text and basic features are listed in the table below.
	1. With the tokenized text features only, the accuracy for a hold-out set of data for our trained model is 0.103.
	2. The accuracy of a model with only the *basic features* is 0.124.
	3. The accuracy improves to 0.134 if the *basic features* of the property are included in the model.  (Basic features are : zip code, home size (sq. ft.), number of bedrooms, number of bathrooms, asking price.)

![Table_accuracy_scores](images/Table_accuracy_scores.png>)


Examples of interesting text features *(More will be added here to elaborate why these features are important, how they impact the price_ratio i.e. higher or lower)*
---------------
- View
![View violin plots](images/view_violin.png>)


- Amaz
![Amaz violin plots](images/amaz_violin.png>)

- Contractor
![Contractor violin plots](images/contractor_violin.png>)

- Opportun
![Opportun violin plots](images/opportun_violin.png>)

- Beach
![Beach violin plots](images/beach_violin.png>)

- Ocean
![Ocean violin plots](images/ocean_violin.png>)



**Conclusions/Summary**
===============
1. A Random Forest Model trained by featurized texts of the an agent's description to predict the sales/asking price ratio has an accuracy of 0.103.  This means 0.13 of the price ratio variance can be explained by the description.

2.  When the basic features of a property (zip code, number of bedrooms, number of bathrooms, home size, and asking price) are included, the model accuracy improves to 0.134.  

3.	A webapp was constructed to allow the user to enter these parameters and estimate the sale price of a home they are intersted in.

Acknowledgments
===============
- Brittany Murphy, Keying Ye for data and analysis.
- Francesco Ciuci for webapp/flask.
