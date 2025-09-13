import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D
import os
# ----------------------------
# Load and prepare data
# ----------------------------
data = pd.read_csv(
    r"Mall_Customers.csv"
)

X = data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train KMeans
kmeans = KMeans(n_clusters=5, random_state=42)
data['Cluster'] = kmeans.fit_predict(X_scaled)

# Train Regression for Spending Score
X_reg = data[['Age', 'Annual Income (k$)']]
y_reg = data['Spending Score (1-100)']
regressor = LinearRegression()
regressor.fit(X_reg, y_reg)

# Cluster labeling
cluster_summary = data.groupby('Cluster')[['Age','Annual Income (k$)','Spending Score (1-100)']].mean()
labels_dict = {}
for idx, row in cluster_summary.iterrows():
    age, income, spending = row
    if age < 35 and income > 60 and spending > 60:
        labels_dict[idx] = 'Young High Income High Spenders'
    elif age < 35 and income <= 60 and spending > 60:
        labels_dict[idx] = 'Young Low Income High Spenders'
    elif age < 35 and spending <= 60:
        labels_dict[idx] = 'Young Average Spenders'
    elif age >= 35 and income > 60 and spending > 60:
        labels_dict[idx] = 'Adult High Income High Spenders'
    elif age >= 35 and spending > 60:
        labels_dict[idx] = 'Adult High Spenders'
    else:
        labels_dict[idx] = 'Adult Average Spenders'

data['Cluster_Label'] = data['Cluster'].map(labels_dict)

# ----------------------------
# Predictor function
# ----------------------------
def predict_customer(age, income):
    # Predict spending score
    spending_pred = regressor.predict([[age, income]])[0]

    # Predict cluster
    scaled_features = scaler.transform([[age, income, spending_pred]])
    cluster = kmeans.predict(scaled_features)[0]
    label = labels_dict.get(cluster, 'Other')

    # 2D Plot
    fig2d, ax2d = plt.subplots(figsize=(6,5))
    sns.scatterplot(data=data, x='Annual Income (k$)', y='Spending Score (1-100)',
                    hue='Cluster_Label', palette='Set1', s=70, ax=ax2d)
    ax2d.scatter(income, spending_pred, color='black', s=150, marker='X', label='New Customer')
    ax2d.set_title("Income vs Spending Score")
    ax2d.legend()

    # 3D Plot
    fig3d = plt.figure(figsize=(7,6))
    ax3d = fig3d.add_subplot(111, projection='3d')
    ax3d.scatter(data['Age'], data['Annual Income (k$)'], data['Spending Score (1-100)'],
                 c=data['Cluster'], cmap='Set1', s=50)
    ax3d.scatter(age, income, spending_pred, color='black', s=150, marker='X', label='New Customer')
    ax3d.set_xlabel("Age")
    ax3d.set_ylabel("Income (k$)")
    ax3d.set_zlabel("Spending Score")
    ax3d.set_title("3D Segmentation")
    ax3d.legend()

    return (
        f"Predicted Spending Score: {spending_pred:.2f}\nCluster: {cluster}\nLabel: {label}",
        fig2d,
        fig3d
    )

# ----------------------------
# Gradio Interface
# ----------------------------
iface = gr.Interface(
    fn=predict_customer,
    inputs=[gr.Number(label="Age", value=30), gr.Number(label="Annual Income (k$)", value=70)],
    outputs=[gr.Textbox(label="Prediction"), gr.Plot(label="2D Plot"), gr.Plot(label="3D Plot")],
    title="Mall Customer Segmentation & Spending Predictor",
    description="Enter Age and Annual Income to predict Spending Score, Cluster, and see plots."
)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))  
    iface.launch(server_name="0.0.0.0", server_port=port)



