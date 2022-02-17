import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn import linear_model as lm

df=pd.read_csv(r"C:\Codes\Python\Data\CarPrice_Assignment_modified.csv")        #Path of Car Dataset
print("PROVIDED DATASET AFTER CLEANING\n\n",df.head(10),"\n\n")

x_train, x_test, y_train, y_test = train_test_split(df[['symboling','fueltype','aspiration','doornumber','carbody','drivewheel',
                                'enginelocation','wheelbase','enginesize','boreratio', 'stroke',
                                'compressionratio','horsepower','peakrpm','citympg','highwaympg']],
                         df[['price']],test_size=0.25)

reg=lm.LinearRegression()
reg.fit(x_train,y_train)

y_predict=reg.predict(x_test)

print("\n\nAccuracy= ",(round(reg.score(x_test,y_test)*100,2)),"%\n\n")

sns.regplot(x = y_test, y = y_predict, scatter_kws = {"color": "blue"},
                                                            line_kws={"color": "red" }, ci=0)

plt.xlabel('Predicted', color='white')
plt.ylabel('Truth', color='white')
plt.show()