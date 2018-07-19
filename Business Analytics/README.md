NBA Hackathon Business Analytics

Outside Data in team_data.csv :

	Market_Size : media market population (in 1000s) (Source: Reddit, Nielson)

	Championships : number of championships franchise has won (includes previous cities of franchise) up until 2016 (Source: Wikipedia)

	Playoffs : times team has made playoffs from 2000-2016 (Source: Wikipedia)

	Twitter : number of Twitter followers (in millions) of each team in 2016 (Source: Complex)


Models Tried:

	Best Performing (validation MAPE of approx 0.25):
	Random Forest
	max_depth=18, random_state=0, criterion='mae', n_estimators=35

	Others Tried:
	Linear Regression
	Decision Tree :
	Naive Bayes
	Support Vector Machine
	Multi-Layer Perceptron
	Ridge Regression
	K Nearest Neighbor Regression


Process in Business.ipynb:

	- preprocessing and aggregation of data
		add features for temporal/seasonal effects and home/away features from given and outside datasets
	- output clean features
