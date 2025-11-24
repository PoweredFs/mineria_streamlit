import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

df = pd.read_excel("data.xlsx")

 
le_tipo = LabelEncoder()
le_programa_consec = LabelEncoder()
le_programa = LabelEncoder()
le_departamento = LabelEncoder()

df["TIPO DE INSCRIPCION"] = le_tipo.fit_transform(df["TIPO DE INSCRIPCION"])
df["PROGRAMA CONSECUTIVO"] = le_programa_consec.fit_transform(df["PROGRAMA CONSECUTIVO"].astype(str))
df["PROGRAMA"] = le_programa.fit_transform(df["PROGRAMA"].astype(str))
df["DEPARTAMENTO"] = le_departamento.fit_transform(df["DEPARTAMENTO"].astype(str))
 
X = df[["PROGRAMA CONSECUTIVO", "PROGRAMA", "DEPARTAMENTO"]]
y = df["TIPO DE INSCRIPCION"]
 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

 
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)



gb = GradientBoostingClassifier(random_state=42)
gb.fit(X_train, y_train)
gb_pred = gb.predict(X_test)
# ---------------------------
# 6. Predicciones sobre test
# ---------------------------
y_pred = model.predict(X_test)

# ---------------------------
# 7. MÉTRICAS
# ---------------------------
print("\n----- MÉTRICAS DEL MODELO RANDOM FOREST-----\n")
print("Accuracy:", accuracy_score(y_test, gb_pred))
print("Precision :", precision_score(y_test, gb_pred, average="macro"))  

print("\n----- MÉTRICAS DEL MODELO GB-----\n")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision :", precision_score(y_test, y_pred, average="macro"))  


print("\n----- PREDICCIÓN POR CONSOLA -----")

# Ingresar valores
prog_consec_input = input("Ingrese PROGRAMA CONSECUTIVO: ")
programa_input = input("Ingrese PROGRAMA: ")
departamento_input = input("Ingrese DEPARTAMENTO: ")
 
prog_consec_encoded = le_programa_consec.transform([prog_consec_input])[0]
programa_encoded = le_programa.transform([programa_input])[0]
departamento_encoded = le_departamento.transform([departamento_input])[0]

# Crear fila para predecir
row = [[prog_consec_encoded, programa_encoded, departamento_encoded]]

# Hacer predicción
pred = model.predict(row)[0]
prediccion_gb = gb.predict(row)[0]

# Decodificar predicción
pred_label = le_tipo.inverse_transform([pred])[0]
pred_label_2 = le_tipo.inverse_transform([prediccion_gb])[0]

print("\nEl TIPO DE INSCRIPCIÓN estimado es (RANDOM FOREST):", pred_label)
print("\nEl TIPO DE INSCRIPCIÓN estimado es (GB):", pred_label_2)