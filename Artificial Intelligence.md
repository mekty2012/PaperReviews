# Natural Language Processing


### VADER: A Parsimonious Rule-based Model for sentiment Analysis of Social Media Text

Topic : Natural Language Processing, Sentiment Analysis

<http://comp.social.gatech.edu/papers/icwsm14.vader.hutto.pdf>

VADER is a data driven rule based model for sentimental analysis. Unlike other approaches which relies on machine learning or neural networks,
VADER uses rules to compute sentiment.
First, VADER consists word dictionaries with their sentiment from -4 to 4. The lexicons are collected from previous works including LIWC, ANEW, GI, and
more acronyms, initialisms, slangs, western-style emoticons are included.
To give sentiment intensity for each lexicons collected, author applied WotC approach. 
Now with lexicon dataset, VADER also analyzed 400 positive/negative tweets that scored top, and extracted five heuristics that affects intensity a lot.
The five heuristics are that, punctuation, capitalization increases intensity, degree modifier are crucial, contrasitive conjuction is important on measurement,
and trigram was enough to catch 90% of intensity flip due to negations.
