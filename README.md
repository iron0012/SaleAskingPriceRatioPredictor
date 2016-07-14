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

This means the tokenized agents’ descriptions by themselves can explain 10.3% of the variance in the predictions.  While this score may seem low at first glance, it is not surprising because the price ratio is determined by many factors which are only partly or not-at-all captured by the agents' descriptions, such as the economy, the buyers' state of mind and finances.  Also, the San Francisco Bay Area market is one that is skewed toward optimism, so it is extra challenging to distinguish the sentiments of the agent.   At what our study shows that the agents' descriptions has some predictive values for the price ratio.  Further work in the future should focus on an area where the housing market is less skewed (i.e., where the average price ratio ~ 1).

Analysis
========
Data
----
- 12,194 properties from 2014-16 in the the counties of San Mateo, San Francisco, and Marin.
- Single family residences only. (Condos and townhouses are excluded, although some slipped through the cracks.)
- Zip code, home size, number of bedrooms, number of bathrooms, and asking prices are included in the training data.
- The average asking price of all properties is $1.26M. The median asking price is $899,000.
- The average price ratio of of all properties is 1.064.  However there is a large concentration at price ratio = 1, indicating that many buyers just pay the asking prices for the properties.
![Distibution of price ratio](images/price_ratio_distribution.png>)

Vectorization of the agents' descriptions of the property.
----
- Employed the Countvectorizer of the scikitlearn library.  (We experimented with TfidfVectorizer but found no improvement)
- Included 1- and 2-grams only.
- Maximum document frequency (max_df) = 0.9, min_df=75.
- Maximun features used per tree, i.e. bagging (max_features) = 0.8
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
'zillow', 'zillow estim', 'zillow valu', 'home', 'bedroom', 'room', 'bathroom'.

Modeling
--------
- We experimented with scikitlearn's RandomForestRegressor and GradientBoostingRegressor at the beginning, but decided to focused on only the RandomForestRegressor because the results between the two are not very different.

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


Selected examples of interesting text features
----------------------------------------------

Here we discuss some examples of text features that we found interesting.  Note that these features have been 'stemmed' and 'lemmatized,' which means they have been converted to the pseudo-roots of the words to facilitate grouping of the same types of words.   

HDWD FLRS', average price ratio = 1.123, occurrence = 64.

- It is well known in San Francisco that hardwood floors are more desirable over carpeted floors.  It is healthier and adds class to the home.  With the average price ratio being 5.5% higher the overall average of 1.064, it represents a $68,000 increase in a property's value on the average.

- Description a property sold for 1.27x asking price:
*"SF Marina Style loved/maintained by family approx. 68 yrs! Period cm+ Ctrl Heat, Brand New Hdwd Flrs, New Kitc Flr, New Toilet & Vanity, Fresh Paint, w/new lite fixtures! UP: 2 bed/1 ba, Frml Entry, Lg Frml Liv Rm w/Frplc, LG Frml Din rm, Cozy Bkfst Rm w/built-in cab, country-style kitchen. DOWN: 3rd bed, HUGE gar/bsmt w/workbench, laundry, storage, garden access. Great Potential!"*

![hdwd flrs violin plots](images/HDWD FLRS_violin_plot.png>)


'TRUST' price ratio = 1.13, occurrence = 181.

- A 'trust sale' usually occurs when the owner passes away.   These properties tend to be priced lower to 'to go'.  For example, the typical asking price for properties with 'trust' in the description is $872,318 which is well below the avearge asking price of $1.27M.  However, since the asking prices are so low, the buyers must be prepared to bid considerably higher than the average 1.064x the asking price.  Our data indicates that typical sale price is 1.13x or almost $100k above asking!

- Description a property sold for 1.27x asking price: *"Successor Trustee Sale sold in AS IS Condition. Owner died in the home of natural causes. Buyer shall pay transfer tax and all cost associated with city/county water, energy, hot water heater and smoke detector ordinances.  Submit fully signed discl packet w/offer using the CAR Purchase Agreement, Trust Advisory and SFAR AS IS Addendum.  Pre-escrow is opened with Gerrie at Chicago on Market Street.  Offers will be reviewed as received after 12PM on Tuesday June 10th.
"*

![Trust violin plots](images/TRUST_violin_plot.png>)

OPPORTUN, price ratio = 1.083, occurrence = 836.

- It appears that when agents describes a property as an 'opportunity', they generally mean it.  On the average, properties with 'OPPORTUN' in the description go for 1.082x the asking price, which is 1.8% higher the avearge.  This translates into about $12k for the avearge house.  

- Desciption of a property that solde for 1.20x the asking price with 'OPPORTUN' in the description:
*"This is the next up and coming area. Don't miss out on the opportunity. Amazingly affordable home with character, fantastic sunny weather, easy commute: only one block from the 3rd St. Rail, 2 blocks from freeways, This home is filled with old world charm &  character not made anymore. The Bay Windows in the spacious living room & elegant bedroom boast a charming decorative bench. Lg eat in kitchen"*

![Opportun violin plots](images/Opportun_violin_plot.png>)

'SHOP RESTAUR', average price ratio = 1.089, occurrence = 244.  
- It is well known that living near shops and restaurants are highly desirable, not just for young professionals but for everyone, so it is not surprising that 'shop restaur' has an above average price ratio of 1.089.  That means an average property that has an asking price of $1.24M can fetch $15k if it has 'SHOP RESTAUR' in its description.

- Description of a property that sold for only 1.48x the asking price: *"Incredible Value in the Heart of Potrero Hill! This Diamond in the Rough has Endless Possibilities! Specious Four Bedrooms with Three Full Baths has The Blank Canvas you've been Looking for. Close to Shopping, Restaurants, Cafes, and an Easy Commute. Though it needs a Little Bit of Everything... It has the Perfect location! Lots of Space to Let Your Imagination Run Wild."*

![shop restaur violin plots](images/SHOP RESTAUR_violin_plot.png>)


BEACH', average price ratio = 1.025, occurrence = 458.
- (See explanation for 'OCEAN' below)

![Beach violin plots](images/BEACH_violin_plot.png>)

'OCEAN', average price ratio = 1.032, occurrence = 567.
- In a hot market like San Francisco, it is difficult to find tokenized text features that indicate a "below avearge" property.  However, here are a couple.  ****eing near a *beach* or the *ocean* would normally be considered desirable, but this is not born out by our data from the Bay Area.  Little known to outsiders, few people go to the beach here.  In fact, the famous San Francisco fog and wind are extra severe at the ocean front.  Moreover, it takes a lot longer to commute from the beach to work in downtown whether by car or by public transportation.  With the average price ratio 3.6% (for 'beach') or 3.0% (for 'ocean') *lower* than the overall average of 1.064, it means a typical property of $1.24M with 'beach' or 'ocean' in the description would decrease $48k or $40k, respectively.

- Description of a property that sold for only 0.85x the asking price:
  *"Calling all Buyers what a great Location (4 Bedrooms 2 Bathrooms) 2BR/1 bath up and 2 BR/1 Bath down in Outer Sunset area! Close to transportation 2blks away and taraval, Blocks away from schools and shopping. Ocean Beach is nearby. What more can you ask for? Great Home! As IS sale, Tenant occupied."*

![Ocean violin plots](images/OCEAN_violin_plot.png>)


**Conclusions/Summary**
===============
1. A Random Forest Model trained by featurized texts of the an agent's description to predict the sales/asking price ratio has an accuracy of 0.103.  This means 0.13 of the price ratio variance can be explained by the description.

2.  When the basic features of a property (zip code, number of bedrooms, number of bathrooms, home size, and asking price) are included, the model accuracy improves to 0.134.  

3.	A webapp was constructed to allow the user to enter these parameters and estimate the sale price of a home they are intersted in.

**Future directions**
===============
In retrospect, we have selected a dataset that is not ideal for price ratio regression analysis.   The San Francisco Bay Area is a very hot housing market, as reflected by the higher-than-average sale/asking price ratio (1.065) and asking price ($1.26M) in our study.   One can surmise that the mood of all the agents was very optimistic and positive, thus making it difficult to apply more profound NLP technique such as sentimental analysis effectively.   Future work should be done on a data set from an area where they is a more equitable market (average sale/asking price ratio ~ 1).

Acknowledgments
===============
- I'd like to acknowledge Brittany Murphy and Keying Ye of Housecanary for their guidance in their analysis, and Francesco Ciuci for setting up the webapp wiht python/flask.
