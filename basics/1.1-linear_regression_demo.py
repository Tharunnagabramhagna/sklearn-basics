''' GENERAL CONCEPT : '''

# Linear Regression :
''' Linear Regression is a supervised machine learning algorithm that is used to predict
a continuous numerical value by learning a linear relationship between the 
input features and a target variable. '''

# continuous numerical value --> E.g. house prices, temperature, sales or salary.
# linear relationship --> Model trys to fit a straight line through the data.

# E.g. problem) Taxi fare

# Original Mathematical Formula --> y = mx + c
# Here, y --> predicted value
#       m --> slope
#       x --> input features
#       c --> intercept

# E.g. Imagine a taxi charges,
# 1) ₹50 as a fixed starting fee. --> c
# 2) ₹15 for every kilometre travelled. --> m
# Let, x be the distance in kilometres and y be the total fare.

# then the equation is , y = 15x + 50

# Explanation : Even if x is 0 the taxi receives 50 rs as starting price so that becomes our
# intercept and also the 15 rs is taken for each km depending on x so 15 rs becomes the slope.

# Q) A person travels 12 km. Calculate the total taxi fare for the above conditions.

# Ans)
# Given , x = 12km
# So, y = 15 * 12 + 50 = 180 + 50 = 230

# therefore, The total Fare = 230 rs

# E.g. problem) Predict the marks based on the time taken for study hours

# Modified ML formula-1 --> ŷ = wx + b
# Here,  ŷ --> predicted value
#        w --> weight (Controls input influence)
#        x --> input features
#        b --> bias (Adjustable prediction offset)

# E.g. w = 10, b = 5 
# then the equation is ŷ = 10x + 5
# Q) Suppose a student studies for 5 hours. What marks does the model predict?
# Ans)    ŷ = 10 * 5 + 5 = 50 + 5 = 55
#         ŷ = 55 marks


# E.g. (problem) --> Advertising and sales prediction

# Modified ML formula-2 --> ŷ = b₀ + b₁x
# Here, ŷ --> Predicted value
#       b₀ --> intercept
#       b₁ --> slope
#       x --> input features

# Imagine a company wants to predict sales based on its advertising budget.
# The training process produces these estimates:
# b₀ = 20
# b₁ = 3
# then the equation is, ŷ = 20 + 3x

# Q) If the company spends ₹10 lakh on advertising, predict its sales.
# Ans)  ŷ = 20 + 3(10) = 20 + 30 = 50 
# therefore, the sales are 50 lakhs

''' Q) Learn the relationship between house area and house price, 
then use that relationship to predict prices for new houses.'''

# Ans)

# House Price Predicator (Linear Regression example)
# Linear Regression using Pure Python

# STEP-1 : Intialize Training Data

areas = [500, 700, 900, 1100, 1300] # areas --> X (input features)
prices = [25, 35, 45, 55, 65] # prices --> Y (target label)


# STEP-2 : Calculate the averages

n = len(areas)

mean_x = sum(areas) / n  # average area
mean_y = sum(prices) / n  # average price

# STEP-3 : Calculate the weight (slope)

numerator = 0
denominator = 0

# formula, w = ∑ (xi - x̄) * (yi - ȳ) / ∑ (xi - x̄) ^ 2
for i in range(n):
    numerator += (areas[i] - mean_x) * (prices[i] - mean_y)
    denominator += (areas[i] - mean_x) ** 2

wt = numerator / denominator

# STEP-4 : Calculate the bias (intercept)

bias = mean_y - (wt * mean_x)

# STEP-5 : Display the trained result

print("MODEL TRAINING COMPLETED!")
print("Weight : ", wt)
print("Bias : ",bias)

# STEP-6 : Predict a new house price

new_area = float(input("\nEnter house area in sq.ft : "))

predicated_price = wt * new_area + bias # y = wx + b

print("Predicated House Price :  ₹", round(predicated_price, 2), "lakhs")