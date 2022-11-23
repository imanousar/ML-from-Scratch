from UniLinearRegression import UniLinearRegression
import numpy as np

# Ensure reproducability
seed = 42
np.random.seed(seed)

x = np.linspace(0, 100, 100)  # training examples
y = 2 * x + 10 * np.random.normal(size=100)  # labels

model = UniLinearRegression()
model.fit(x, y)
y_pred = model.predict(x)

for i in range(len(y)):
    print(f'{y_pred[i]}: {y[i]}')
