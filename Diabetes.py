import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar el dataset
data_url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
columns = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]
data = pd.read_csv(data_url, header=None, names=columns)

# 1. Exploración de los datos
print("Primeros 5 registros del dataset:")
print(data.head())

# Resumen estadístico y chequeo de valores faltantes
print("\nResumen del dataset:")
print(data.describe())
print("\nValores faltantes por columna:")
print(data.isnull().sum())

# Visualización de distribuciones
sns.pairplot(data, hue="Outcome", diag_kind="kde")
plt.suptitle("Distribuciones por clase", y=1.02)
plt.show()

# Correlaciones
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
plt.title("Mapa de calor de correlaciones")
plt.show()

# 2. Preprocesamiento
# Reemplazar ceros por NaN en columnas relevantes (valores imposibles)
columns_to_clean = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
for col in columns_to_clean:
    data[col] = data[col].replace(0, np.nan)

# Rellenar valores faltantes con la mediana de cada columna
data.fillna(data.median(), inplace=True)

# Separar características y etiquetas
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# Escalar características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Dividir datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# 4. Entrenamiento y comparación de modelos
models = {
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(probability=True, random_state=42),
    "MLP": MLPClassifier(max_iter=300, random_state=42)
}

results = {}
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    print(f"\nEntrenando modelo: {name}")
    scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring='roc_auc')
    print(f"AUC-ROC (Validación Cruzada): {scores.mean():.3f} (+/- {scores.std():.3f})")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("\nReporte de Clasificación:")
    print(classification_report(y_test, y_pred))

    auc = roc_auc_score(y_test, y_proba)
    print(f"AUC-ROC en Test: {auc:.3f}")

    results[name] = {
        "model": model,
        "roc_auc": auc,
        "conf_matrix": confusion_matrix(y_test, y_pred)
    }

# 5. Visualización de Resultados
for name, res in results.items():
    print(f"\n{name}: Matriz de Confusión")
    sns.heatmap(res["conf_matrix"], annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"Matriz de Confusión - {name}")
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.show()

# Comparación final de modelos
roc_aucs = {name: res['roc_auc'] for name, res in results.items()}
plt.bar(roc_aucs.keys(), roc_aucs.values(), color=['blue', 'green', 'orange'])
plt.title("Comparación de Modelos - AUC-ROC")
plt.ylabel("AUC-ROC")
plt.show()