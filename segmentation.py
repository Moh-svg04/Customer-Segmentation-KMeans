import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Chargement des données
url = "https://raw.githubusercontent.com/vjchoudhary7/Customer-Segmentation-Tutorial-in-Python/master/Mall_Customers.csv"
df = pd.read_csv(url)
df.columns = ['ID', 'Genre', 'Age', 'Revenu_Annuel', 'Score_Depenses']

# Préparation des données
X = df[['Revenu_Annuel', 'Score_Depenses']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Clustering
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
df['Segment'] = kmeans.fit_predict(X_scaled)

# Visualisation (Optionnel pour le script, mais important pour le résultat)
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Revenu_Annuel', y='Score_Depenses', hue='Segment', palette='viridis')
plt.title('Segmentation des Clients')
plt.savefig('resultat.png') # Sauvegarde l'image
