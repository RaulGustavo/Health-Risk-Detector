import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.patches import Circle

df_stroke = pd.read_csv("./datasets/brain_stroke.csv")
stroke_1 = df_stroke.head(100)
stroke_2 = df_stroke.tail(100)
stroke_100 = pd.concat([stroke_1, stroke_2])

df_heart = pd.read_csv("./datasets/heart.csv")
heart_1 = df_heart.head(200)
heart_2 = df_heart.tail(200)
heart_100 = pd.concat([heart_1, heart_2])

def strokeKNN():
    X = np.array(stroke_100[['age', 'bmi', 'heart_disease']])
    y = np.array(stroke_100['stroke'])
    
    knn = KNeighborsClassifier(n_neighbors=10)
    knn.fit(X, y)

    input_age = int(input("Age: "))
    input_weight = float(input("Weight (kg): "))
    input_height = float(input("Height (m): "))
    input_hd = int(input("Heart Disease?: "))

    height_sq = input_height**2

    input_bmi = input_weight/height_sq

    print("\nBMI: ", input_bmi)

    patient = np.array([[input_age, input_bmi, input_hd]])

    prediction = knn.predict(patient)

    if prediction[0] == 0:
        print("\nThe pacient is healthy and not at risk\n\n")
    elif prediction[0] == 1:
        print("\nThe pacient is at risk\n\n")
        print("WE RECOMMEND YOU START THE FOLLOWING HABITS")
        print("1. Manage Blood Pressure")
        print("2. Control Blood Sugar")
        print("3. Maintain a Healthy Weight")
        print("4. Stay Physically Active")
        print("5. Avoid Smoking")
        print("6. Limit Alcohol Intake")
        print("7. Manage Stress\n\n")
        
    distance, index = knn.kneighbors(patient)

    plt.figure(figsize=(8, 6))

    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='blue', label='Healthy', marker='^')
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='red', label='At Risk', marker='o')

    plt.scatter(patient[:, 0], patient[:, 1], color="black", label='Patient', s=200, marker='*')  # Larger marker for patient

    radio = np.max(distance)
    circulo = Circle(patient[0], radio, color='black', fill=False, linestyle='--', label='Radio k')
    plt.gca().add_patch(circulo)

    plt.xlabel('Age')
    plt.xlim(0, 85)
    plt.ylabel('BMI')
    plt.ylim(15, 50)
    plt.title('Stroke Detection')
    plt.legend()
    plt.grid(True)
    plt.show()
        
    
def heartKNN():
    X = np.array(heart_100[['Age', 'Cholesterol', 'RestingBP']])
    y = np.array(heart_100['HeartDisease'])

    knn = KNeighborsClassifier(n_neighbors=10)

    knn.fit(X, y)

    input_age = int(input("Age: "))
    input_chol = float(input("Cholesterol Level: "))
    input_bp = float(input("Resting Blood Pressure (SBP): "))

    patient = np.array([[input_age, input_chol, input_bp]])

    prediction = knn.predict(patient)

    if prediction[0] == 0:
        print("\nThe pacient is healthy and not at risk\n\n")
    elif prediction[0] == 1:
        print("\nThe pacient is at risk\n\n")
        print("WE RECOMMEND YOU START THE FOLLOWING HABITS")
        print("1. Adopt a Heart-Healthy Diet: Limit saturated and trans fats, sugar and salt")
        print("2. Stay Physically Active with low-impact aerobic exercise")
        print("3. Monitor and Control Blood Pressure")
        print("4. Stay Hydrated")
        print("5. Quit Smoking")
        print("6. Manage Stress")
        print("7. Get Adequate Sleep\n\n")

    distance, index = knn.kneighbors(patient)

    plt.figure(figsize=(8, 6))

    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='blue', label='Healthy', marker='^')
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='red', label='At Risk', marker='o')

    plt.scatter(patient[:, 0], patient[:, 1], color="black", label='Patient', s=200, marker='*')  # Larger marker for patient

    radio = np.max(distance)
    circulo = Circle(patient[0], radio, color='black', fill=False, linestyle='--', label='Radio k')
    plt.gca().add_patch(circulo)

    plt.xlabel('Age')
    plt.xlim(0, 100)
    plt.ylabel('Cholesterol Level')
    plt.ylim(100, 400)
    plt.title('Heart Disease Detection')
    plt.legend()
    plt.grid(True)
    plt.show()

while True:
    print("MENÚ")
    print("1.- Stroke")
    print("2.- Heart Disease")
    print("3.- Exit")
    opc = int(input("Choose an option: "))
    if opc == 1:
        strokeKNN()
    if opc == 2:
        heartKNN()
    if opc == 3:
        print("\nClosing app...")
        break
    
