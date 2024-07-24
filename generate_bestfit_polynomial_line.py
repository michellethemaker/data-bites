import numpy as np
import matplotlib.pyplot as plt

# sample data
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([2.3, 5.1, 5.9, 7.2, 10.5, 12.9, 15.5, 19.8, 22.3, 19.7])


# fit to xth-order polynomial, generate polynom fn
degree = input("input the degree: ")    
# degree = 4
try:
    degree_float = float(degree)
    if not degree_float.is_integer and degree_float < 0:
        raise ValueError(f'ERROR: {degree_float} is invalid; Make sure degree is positive integer.')
except ValueError as e:
    print(f'ERROR HERE: , {e}')
    exit(1)
degree = int(degree)
coefficients = np.polyfit(x, y, degree)

polynomial = np.poly1d(coefficients)

# generate values for plotting fit
x_fit = np.linspace(min(x), max(x), 100)
y_fit = polynomial(x_fit)

# plot data and fit
plt.scatter(x, y, color='r', label='Data Points')
plt.plot(x_fit, y_fit, color='b', label=f'{degree}th order poly fit')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.title(f'{degree}th ord polynom best fit curve')
plt.savefig(f'bestfit{degree}.png')
plt.show()

# print polynom eq
print(f'polynom coefficients (highest degree first): \n{coefficients}')
print(f'polynom function: \n{polynomial}')
