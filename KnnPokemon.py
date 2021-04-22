import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# Read Pokemon Data csv file
df = pd.read_csv("pokemon.csv")
df = df[df.is_legendary != 1]
df = df.drop(columns=['abilities', 'is_legendary', 'generation', 'weight_kg',  'attack', 'base_egg_steps', 'base_happiness', 'classfication', 'base_total', 'capture_rate', 'defense',
                      'experience_growth', 'height_m', 'hp', 'japanese_name', 'name', 'percentage_male', 'pokedex_number', 'sp_attack', 'sp_defense', 'speed'])

# Get x and y values
y = df.type1.values
df.drop(columns=['type1', 'type2'], inplace=True)
x = df.values

# Get train and test values
xtrain, xtest, ytrain, ytest = train_test_split(x,y, test_size = 0.2, random_state=1, stratify=y)

# Create and Fit Model
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(xtrain, ytrain)

# Returns the score of model based on K value
def returnScore(k, xtrain, xtest, ytrain, ytest):
  knn = KNeighborsClassifier(n_neighbors=k)
  knn.fit(xtrain, ytrain)
  return knn.score(xtest, ytest)

result = [*map(lambda i:returnScore(i,xtrain, xtest, ytrain, ytest), range(1,25))]

# Print Graph of K's
plt.plot(result)
# Print best K value
print('Best Value of K:',np.argmax(result) + 1 )