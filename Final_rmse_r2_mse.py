import numpy as np
import pandas as pd
import pandas
import matplotlib.pyplot as plt
data = pandas.read_excel('sampledata.xlsx')
print(data.shape)
data.head()

# Collecting X and Y
X = data['lift1_True'].values
Y = data['lift2_Predict'].values
 
# Mean X and Y
mean_x = np.mean(X)
mean_y = np.mean(Y)

# Total number of values
m = len(X)

# Using the formula to calculate b1 and b2
numer = 0
denom = 0
for i in range(m):
    numer += (X[i] - mean_x) * (Y[i] - mean_y)
    denom += (X[i] - mean_x) ** 2
b1 = numer / denom
b0 = mean_y - (b1 * mean_x)

# Print coefficients
print(b1, b0)

# Calculating Root Mean Squares Error
rmse = 0
for i in range(m):
    y_pred = b0 + b1 * X[i]
    rmse += (Y[i] - y_pred) ** 2
rmse = np.sqrt(rmse/m)
print(rmse)
# Calculating  Mean Squares Error MSE
mse = 0
for i in range(m):
    y_pred = b0 + b1 * X[i]
    mse += (Y[i] - y_pred)
    mse=mse/m
print(mse)
# calculating r2
ss_t = 0
ss_r = 0
for i in range(m):
    y_pred = b0 + b1 * X[i]
    ss_t += (Y[i] - mean_y) ** 2
    ss_r += (Y[i] - y_pred) ** 2
r2 = 1 - (ss_r/ss_t)
print(r2)

# The coefficients
print('Coefficients: \n', b1,b0)
# The root  mean squared error
print('ROOT Mean squared error: \n',rmse)
# The root  mean squared error
print('Mean squared error: \n',mse)      
# Explained variance score: 1 is perfect prediction
print('Coefficent of determination : ',r2)

# Plot outputs
plt.scatter(X, Y,  color='black')
plt.plot(X, Y, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.xlabel('list1')
plt.ylabel('list2')
# plt.legend()
plt.show()

