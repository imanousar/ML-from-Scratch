import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from mlfromscratch.src.supervised_learning.UniLinearRegression import \
    UniLinearRegression

# Title
st.title("Linear Regression")
st.write("Here is an example of a Linear Regression model.")


# Sidebar
st.sidebar.markdown("## Controls")
st.sidebar.markdown("You can **change** the values to change the *plot*.")

x_start = st.sidebar.slider('X: Start', min_value=0,
                            max_value=20, value=0, step=1)
x_end = st.sidebar.slider('X: End', min_value=50,
                          max_value=200,  value=100, step=1)
samples = st.sidebar.slider('X: Samples', min_value=50,
                            max_value=500,  value=100, step=1)
slope = st.sidebar.slider('Slope', min_value=-5.0,
                          max_value=5.0,  value=2.0, step=0.01)
epochs = st.sidebar.slider('Epochs', min_value=0,
                           max_value=40,  value=20, step=1)
lr = st.sidebar.slider('Learning rate', min_value=0.00001,
                       max_value=0.001,  value=0.0001, step=0.00001,
                       format="%f")


# Model Config

x = np.linspace(x_start, x_end, samples)  # training examples
y = slope * x + 10 * np.random.normal(size=samples)  # labels

with st.echo():
    model = UniLinearRegression(epochs=epochs, learning_rate=lr)
    model.fit(x, y)
    y_pred = model.predict(x)

# Plot
fig = plt.figure(figsize=(5, 3), dpi=20)
ax = plt.subplot(111)
plt.scatter(x, y)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Linear Regression')
ax.plot(x, y_pred, c='#ff7f0e')

st.pyplot(fig)
