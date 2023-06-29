# Forecasting Crude Oil Price

# In[321]:
import streamlit as st
import pandas as pd
from prophet import Prophet
# importing libraries
import pandas as pd
import numpy as np
# to Visualize
import seaborn as sns
import matplotlib.pyplot as plt
from pandas import Grouper
from pandas import DataFrame
from pandas.plotting import lag_plot
import statsmodels.graphics.tsaplots as tsa_plots
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings
warnings.filterwarnings('ignore')
# For stationarity check
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
import pickle
# In[150]:
st.markdown('''
<style>
.stApp {
    
    background-color:#8DC8ED;
    align:center;\
    display:fill;\
    border-radius: false;\
    border-style: solid;\
    border-color:#000000;\
    border-style: false;\
    border-width: 2px;\
    color:Black;\
    font-size:15px;\
    font-family: Source Sans Pro;\
    background-color:#8DC8ED;\
    text-align:center;\
    letter-spacing:0.1px;\
    padding: 0.1em;">\
}
.sidebar {
    background-color: black;
}

.st-b7 {
    color: #8DC8ED;
}
.css-nlntq9 {
    font-family: Source Sans Pro;
}
</style>
''', unsafe_allow_html=True)


data=pd.read_excel('Crude_oil_price.xlsx',parse_dates=True,squeeze=True)


data['Date'] = pd.to_datetime(data['Date'])

data['Year'] = data['Date'].dt.year


# Treat outliers using winsorization
q1 = data['Price'].quantile(0.25)
q3 = data['Price'].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

data['Price'] = data['Price'].clip(lower=lower_bound, upper=upper_bound)

# In[332]:

# Treat outliers using IQR method
plt.figure(figsize=(12,8))
q1 = data['Price'].quantile(0.25)
q3 = data['Price'].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

data['Treated_Price'] = data['Price'].clip(lower=lower_bound, upper=upper_bound)


#data['first_diff']=data['Treated_Price']-data['Treated_Price'].shift(1)

#data['Year'] = pd.to_datetime(data['Date']).dt.strftime("%Y")
data['Month'] = pd.to_datetime(data['Date']).dt.strftime('%b')
data['Day'] = pd.to_datetime(data['Date']).dt.strftime("%d")


#OHE
month_dummies = pd.DataFrame(pd.get_dummies(data['Month']))
data = pd.concat([data,month_dummies],axis = 1)

data=data.drop(['Month','Price'],axis=1)
data=data.dropna()

# Rename the columns to match Prophet's requirements
data = data.rename(columns={'Date': 'ds', 'Treated_Price': 'y'})
#data.head()
# Create a new Prophet instance
model_pro = Prophet()

# Fit the model to the data
model_pro.fit(data)

# Define the number of periods to forecast
future_periods = 10

# Generate future dates
future_dates = model_pro.make_future_dataframe(periods=future_periods)

# Perform the forecast
forecast = model_pro.predict(future_dates)

# Extract the actual values from the dataset
actual_values = data['y'].values[-future_periods:]

# Extract the predicted values from the forecast
predicted_values = forecast['yhat'].values[-future_periods:]

# Calculate RMSE
rmse_pro = np.sqrt(mean_squared_error(actual_values, predicted_values))

# Print RMSE
print("RMSE:", rmse_pro)





# Prophet model

filename = 'Forcasting_Prophet.pkl'
pickle.dump(model_pro, open(filename, 'wb'))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet

# Load the dataset
data = pd.read_excel('Crude_oil_price1.xlsx')
data = data.rename(columns={'Date': 'ds','Treated_Price': 'y'})

# Create a new Prophet instance
model_pro = Prophet()
model_pro.fit(data)

# Define a function to perform the forecast
def perform_forecast(data, future_periods):
    # Generate future dates
    future_dates = model_pro.make_future_dataframe(periods=future_periods)

    # Perform the forecast
    forecast_result = model_pro.predict(future_dates)

    # Extract the forecasted values
    forecast_values = forecast_result[['ds', 'yhat']].rename(columns={'yhat': 'Forecast'})

    return forecast_values

def main():
    st.title("Crude Oil Price Forecasting")

    # Specify the number of days to forecast using a slider
    future_periods = st.slider('Number of days to forecast price from 29 Dec 2018', min_value=1, max_value=365, value=30, step=1)

    # Perform the forecast
    forecast_data = perform_forecast(data, future_periods)

    # Display the forecasted results
    st.subheader(f"Forecasted Crude Oil Prices for the next {future_periods} days:")
    st.dataframe(forecast_data.iloc[28:])

    # Plot the forecasted values
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(forecast_data['ds'].iloc[28:], forecast_data['Forecast'].iloc[28:], label='Forecast', color='orange')
    #ax.plot(data['ds'].iloc[28:28+future_periods], data['y'].iloc[28:28+future_periods], label='Actual')
    ax.set_xlabel('Date')
    ax.set_ylabel('Crude Oil Price')
    ax.legend()
    st.subheader('Forecasted Graph')
    st.pyplot(fig)

if __name__ == '__main__':
    main()



