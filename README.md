# Flight-Delay-Prediction-using-Two-Stage-Machine-Learning-Engine

This project builds a two-stage machine learning pipeline to predict flight delays, especially those caused by weather, using classification and regression models. The final pipeline combines **XGBoost Classification** and **Extra Trees Regression**, achieving an accuracy of **96.20%**.

---

## 📁 Project Structure

<pre>
📦flight-delay-prediction/
┣ 📂data/ # Merged flight and weather datasets
┣ 📂models/ # Saved ML models
┣ 📂notebooks/ # Exploratory and training notebooks
┣ 📜pipeline.py # Final ML pipeline script
┣ 📜README.md # Project documentation
┗ 📜requirements.txt # Python dependencies
</pre>


---


---

<details>
<summary>🧩 <strong>Problem Statement & Overview</strong></summary>

### 🎯 Objective:
Build a **two-stage predictive ML engine** to forecast the on-time performance of U.S. domestic flights.

### 📚 Dataset Sources:
- **Flight data (2016–2017)** – [Link to flight data](#)
- **Weather data (2016–2017)** – [Link to weather data](#)

### ✈️ Airports Considered:
`ATL, CLT, DEN, DFW, EWR, IAH, JFK, LAS, LAX, MCO, MIA, ORD, PHX, SEA, SFO`

### 🕐 Duration:
- Module 1 (Preprocessing): 20 days
- Module 2 (Classification): 15 days
- Module 3 (Regression): 15 days

### 🔗 Reference Tools:
- [NumPy workbook](#), [Pandas workbook](#), [JSON workbook](#)
- Lookup: Dask, NumPy, Pandas, DateTime, `os`, JSON

</details>

<details>
<summary>🧹 <strong>Module 1: Data Preprocessing</strong></summary>

- Merged flight and weather data by matching datetime and airport fields.
- Filtered only for 15 specified airports and the years 2016 and 2017.

**✅ Recommended Flight Columns:**
- `FlightDate`, `Quarter`, `Year`, `Month`, `DayofMonth`, `DepTime`, `DepDel15`, `CRSDepTime`, `DepDelayMinutes`, `OriginAirportID`, `DestAirportID`, `ArrTime`, `CRSArrTime`, `ArrDel15`, `ArrDelayMinutes`

**✅ Recommended Weather Columns:**
- `WindSpeedKmph`, `WindDirDegree`, `WeatherCode`, `precipMM`, `Visibility`, `Pressure`, `Cloudcover`, `DewPointF`, `WindGustKmph`, `tempF`, `WindChillF`, `Humidity`, `date`, `time`, `airport`

**📉 Techniques Used:**
- Feature selection using correlation matrix
- Feature scaling with `StandardScaler`
- Dimensionality reduction via `PCA` (reduced to 30 principal components)
</details>

<details>
<summary>📊 <strong>Module 2: Classification</strong></summary>

### 🎯 Goal: Predict whether a flight will be delayed.

**Algorithms Used:**
- Logistic Regression
- Decision Tree
- Extra Trees
- XGBoost
- Random Forest

**⚖️ Class Imbalance Handling:**
- Applied **SMOTE** to generate synthetic samples for delayed flights

**📈 Best Model:** `XGBoost Classifier`
- **Accuracy**: 86.85%

| Model             | Accuracy |
|------------------|----------|
| Logistic Regression | 86.56%  |
| Decision Tree       | 81.81%  |
| Extra Trees         | 83.56%  |
| **XGBoost**         | **86.85%**  |
| Random Forest       | 85.97%  |

</details>

<details>
<summary>📈 <strong>Module 3: Regression</strong></summary>

### 🎯 Goal: Predict delay duration in minutes for delayed flights.

**Models Used:**
- Linear Regression
- Extra Trees Regressor
- XGBoost Regressor
- Random Forest Regressor

**📈 Best Model:** `Extra Trees`
- **R² Score**: 0.9687
- **MAE**: 5.96 min
- **MSE**: 109.70

| Model             | MAE   | MSE    | R² Score |
|------------------|--------|---------|----------|
| Linear Regression | 8.17   | 167.05  | 0.9523   |
| **Extra Trees**   | **5.96** | **109.70** | **0.9687** |
| XGBoost           | 6.01   | 145.44  | 0.9585   |
| Random Forest     | 5.96   | 109.39  | 0.9688   |

</details>

<details>
<summary>🔀 <strong>Pipeline Architecture</strong></summary>

A **two-stage ML pipeline** integrates:
1. `XGBoost Classifier` to detect delays
2. `Extra Trees Regressor` to estim

</details>

<details>
<summary>📊 <strong>Range-Based Regression Analysis</strong></summary>

Analyzed delay ranges to fine-tune predictions:

| Delay Range | Frequency | MAE   | MSE   | R² Score |
|-------------|-----------|-------|--------|----------|
| 15–199 min  | 283,217   | 8.91  | 146.82 | 0.9079   |
| 200–399     | 10,207    | 12.61 | 311.63 | 0.8794   |
| 400–599     | 1,101     | 13.99 | 481.43 | 0.8508   |
| 600–799     | 356       | 11.51 | 234.60 | 0.9328   |
| 800+        | 386       | 13.52 | 345.16 | 0.9907   |

Insights:
- Higher delay ranges are easier to predict.
- Short delays are frequent but show more variability.

</details>

<details>
<summary>✅ <strong>Conclusion & Future Work</strong></summary>

### ✅ Achievements:
- Accurate prediction pipeline (96.20% overall performance)
- Addressed imbalance with SMOTE
- Captured nonlinear delay patterns via ensemble models

### 🔮 Future Enhancements:
- Integrate **real-time weather forecasts**
- Add **turbulence prediction**
- Use the system for **dynamic re-scheduling** to reduce flight risks and delays
</details>

---

## 🚀 Getting Started

```bash
# Clone the repository
git clone https://github.com/yourusername/flight-delay-prediction.git
cd flight-delay-prediction

# Install dependencies
pip install -r requirements.txt

# Run pipeline
python pipeline.py

