# Predicting-Users-Votes
The attached TSV file contains information about 146,646 pictures posted at a major Russian railfanning website. (No need to click!) The columns, separated by TABs, are explained in the table:

etitle - picture category
region - region where the picture has been taken
takenon - the date when the picture was taken, represented as YYYY-MM-DD; if the day or months are not known, 00 is used instead; some values in this column may be missing
votedon - the date when the picture was posted to the site; some values in this column may be missing
author_id - the ID of the author, represented as a positive integer number
votes - the number of (up)votes for the picture
viewed - the number of times the pictures was viewed
n_comments - the number of comments to the picture.
Some values in the last three columns may be negative. Treat the negative numbers as NAs.

You are to write a script that will predict the number of upvotes based on any other information from the table, using any predictive model. You will possibly need:

Split the table into the training and testing parts.
Convert some or all date-based columns to Pandas DateTime.
Convert some or all categorical columns to dummies.
Select important features.
Choose a predictive model.
Fit the model and assess the fit quality.
Cross-validate the model.
Repeat steps 4-7, if necessary.
Your dataset has 70% of rows of the complete dataset. I have the remaining 30%. I will evaluate the quality of your model by applying it to my part of the dataset. You will earn a passing grade if the score of the model is at least 51% (better than tossing a coin).

