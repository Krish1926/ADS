import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from lmfit import Model

def read_and_process_data(file_path):
    """
            Read and clean the dataset.

            Parameters:
            - file_path (str): The path to the CSV file.

            Returns:
            - original_data (pd.DataFrame): Original dataset.
            - transposed_data (pd.DataFrame): Transposed dataset.
    """
    # Read the dataset
    original_data = pd.read_csv(file_path)

    # Display unique values in each column
    for column in original_data.columns:
        unique_values = original_data[column].unique()
        print(f"Unique values in {column}: {unique_values}")

    # Replace non-numeric values with NaN
    cleaned_data = original_data.apply(pd.to_numeric, errors='coerce')

    # Replace NaN values with median
    cleaned_data = cleaned_data.fillna(cleaned_data.median())

    # Transpose the data
    transposed_data = original_data.transpose()

    return original_data, cleaned_data, transposed_data


def clustering():
    """
            Apply k-means clustering to the selected columns
            and plot the results.

            Returns:
            None
    """
    normalized_data = (cleaned_data[columns_for_clustering] -
                       cleaned_data[columns_for_clustering].mean()) / \
                      cleaned_data[
                          columns_for_clustering].std()

    kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
    cleaned_data['Cluster'] = kmeans.fit_predict(normalized_data)

    # Plot clustering graph with cluster centers
    plt.scatter(
        cleaned_data[
            'Debt service (PPG and IMF only, % of exports of goods, services and primary income) [DT.TDS.DPPF.XP.ZS]'],
        cleaned_data['Public and publicly guaranteed debt service (% of GNI) [DT.TDS.DPPG.GN.ZS]'],
        c=cleaned_data['Cluster'], cmap='viridis', label='Clusters'
    )
    plt.scatter(kmeans.cluster_centers_[:, 1], kmeans.cluster_centers_[:, 2],
                s=200, marker='X', c='red',
                label='Cluster Centers')
    plt.xlabel('Debt Service (% of exports of goods, services and primary income)')
    plt.ylabel('Public and publicly guaranteed debt service (% of GNI)')
    plt.title('Clustering Results with Cluster Centers')
    plt.legend()
    plt.show()

    # Calculate Silhouette Score
    silhouette_avg = silhouette_score(normalized_data, cleaned_data['Cluster'])
    print(f"Silhouette Score: {silhouette_avg}")


def curveFitting():
    """
            Perform exponential growth curve fitting, plot the results,
            and predict values for specific countries.

            Returns:
            None
    """
    # Extract relevant data
    time_data = cleaned_data['Time']
    debt_service_data = \
        cleaned_data['Public and publicly guaranteed debt service (% of GNI) [DT.TDS.DPPG.GN.ZS]']

    # Define the modified exponential growth model
    def exponential_growth_model(x, a, b):
        return a * np.exp(b * np.array(x))

    # Create an lmfit Model
    model = Model(exponential_growth_model)

    # Set initial parameter values
    params = model.make_params(a=1, b=0.001)

    # Fit the model to the data
    result = model.fit(debt_service_data, x=time_data, params=params)

    # Print fit results
    print(result.fit_report())

    # Plot the data and the fitted curve with confidence interval
    plt.scatter(time_data, debt_service_data, label='Actual Data')
    plt.plot(time_data, result.best_fit, label='Exponential Growth Fit')
    plt.fill_between(time_data, result.eval_uncertainty(), -result.eval_uncertainty(),
                     color='gray', alpha=0.2,
                     label='95% Confidence Interval')
    plt.xlabel('Time')
    plt.ylabel('Public and publicly guaranteed debt service (% of GNI)')
    plt.title('Curve Fit for Debt Service Over Time')
    plt.legend()
    plt.show()

    # Generate time points for prediction
    future_years = [2024, 2025, 2026]

    # Predict values for the future years using the fitted model
    predicted_values = result.eval(x=np.array(future_years))

    # Display the predicted values
    for year, value in zip(future_years, predicted_values):
        print(f"Predicted value for {year}: {value:.2f}")


# Specify the file path
file_path = "Data1.csv"

# Call the method to read and process the data
original_data, cleaned_data, transposed_data = read_and_process_data(file_path)

# Display original, cleaned, and transposed data (optional)
print("Original Data:")
print(original_data.head())

print("\nCleaned Data:")
print(cleaned_data.head())

print("\nTransposed Data:")
print(transposed_data.head())

# Apply k-means clustering
columns_for_clustering = [
    "Forest area (% of land area) [AG.LND.FRST.ZS]",
    "Debt service (PPG and IMF only, % of exports of goods, services and primary income) [DT.TDS.DPPF.XP.ZS]",
    "Public and publicly guaranteed debt service (% of GNI) [DT.TDS.DPPG.GN.ZS]",
    "Present value of external debt (% of GNI) [DT.DOD.PVLX.GN.ZS]"
]

clustering()
curveFitting()


