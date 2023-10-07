import MultipleRegression as mp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

videogames = pd.read_csv("videogames.csv")
videogames.head()

Y_TRAIN = videogames['jp_sales'].values

videogames = videogames.drop(['id', 'jp_sales'], axis=1)
col_to_del = ['platform', 'genre', 'publisher', 'year']
one_hot_enc = pd.get_dummies(videogames, columns=col_to_del)
#one_hot_enc['square_na_sales'] = (one_hot_enc['na_sales'].values)**2
#one_hot_enc['square_eu_sales'] = np.sqrt(one_hot_enc['eu_sales'].values)
#one_hot_enc['log_eu_sales'] = np.log(one_hot_enc['eu_sales'].values)

X_TRAIN = one_hot_enc.values

#print(one_hot_enc.head())

regressor = mp.MultipleRegression(0.001, 5000)

regressor.fit(X_TRAIN, Y_TRAIN)

errs_mae, steps = regressor.getMaeErrs()
errs_mape, steps = regressor.getMapeErrs()
plt1 = plt.figure().subplots()
plt1.plot(steps, errs_mae)
plt2 = plt.figure().subplots()
plt2.plot(steps, errs_mape)

plt.show()

print(Y_TRAIN)
print(regressor.predict(X_TRAIN))
