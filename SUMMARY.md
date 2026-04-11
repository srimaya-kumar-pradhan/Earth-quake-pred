# 🌍 QuakeAI: Earthquake Intelligence & Magnitude Prediction System

### 📊 PART 1: DATASET INFORMATION
*   **Dataset Name:** Historical Indian Earthquake Records (2019-2021)
*   **Dataset Size:** 2,719 records  
*   **Features:** 26 (Engineered from raw geospatial and temporal data)
*   **Target:** `magnitude` (Richter Scale)
*   **Data Types:** Numerical (Geosat), Categorical (Regions), Time-Series (Origin Time)
*   **Preprocessing:** 
    *   **Feature Engineering:** Derived cyclical time features (Sin/Cos), rolling seismic averages, and depth-category flags.
    *   **Cleaning:** Validated coordinate ranges and handled 0.0 depth values using median imputation.
    *   **Scaling:** Applied Log-transform to depth to handle exponential distribution.

---

### 📈 PART 2: TRAINING & TESTING DETAILS
*   **Training Set:** 80% (2,175 samples)  
*   **Testing Set:** 20% (544 samples)  
*   **Splitting Method:** **Time-Series Split** (Sequential)
*   **Justification:** Earthquake data is temporal. We use past data to predict future events, preventing "look-ahead bias" where the model accidentally learns from future data.

---

### 🤖 PART 3: MODEL INFORMATION
*   **Model Used:** **XGBoost Regressor** (Extreme Gradient Boosting)
*   **Why XGBoost?** High performance with tabular data, built-in handling of missing values, and excellent ability to capture non-linear seismic patterns.
*   **Key Hyperparameters:**
    *   `n_estimators`: 800
    *   `learning_rate`: 0.03
    *   `max_depth`: 8
*   **Top Feature Importance:**
    1.  **Depth Log (25%)**: The strongest indicator of magnitude.
    2.  **Region Average Magnitude (12%)**: Historical regional context.
    3.  **Latitude/Longitude Interaction**: Geographical hotspots.

---

### 📊 PART 4: PERFORMANCE METRICS
*   **RMSE: 0.5603** → On average, predictions are within 0.56 magnitude of the actual value.
*   **MAE: 0.4206** → The typical prediction error is very small (0.42).
*   **R² Score: 0.488** → The model explains **48.8% of the variance** in earthquake intensity—a significant result given the stochastic (random) nature of seismic events.

---

### 📊 PART 5: VISUAL INSIGHTS
*   **Regional Distribution:** The **Northeast** and **Himalayan Belt** account for over 45% of total activity (High-risk zones).
*   **Magnitude Trends:** Most events cluster between **3.0 and 4.5 magnitude**; extreme events (6.0+) are rare but detectable.
*   **Depth Relationship:** Shallow earthquakes (≤ 10km) show higher frequency, while deep-focus events (> 100km) are localized in specific subduction zones.
*   **Anomaly Detection:** Uses Z-Score analysis to flag events that significantly deviate from regional norms.

---

### ⚙️ PART 6: SYSTEM ARCHITECTURE
*   **Frontend:** HTML5, Vanilla CSS, **Chart.js** (Analytics), **Leaflet.js** (Interactive Mapping).
*   **Backend:** **Flask (Python)** API for real-time inference and data serving.
*   **ML Engine:** **XGBoost** model loaded via Joblib for sub-second predictions.
*   **Execution Flow:**
    `User Input` → `Flask API` → `Dynamic Feature Engineering` → `XGBoost Prediction` → `JSON Output` → `UI Visualization`

---

### 🚀 PART 7: KEY FEATURES
*   **Live Prediction:** Predicts magnitude based on region, depth, and time.
*   **Dynamic Analytics:** Real-time charts showing seismic trends and frequency.
*   **Interactive Heatmap:** Visualizes historical earthquake clusters across the Indian subcontinent.
*   **Health Monitoring:** Built-in API status and model metric tracking.

---

### ⚠️ PART 8: LIMITATIONS
*   **No Real-time Feed:** Currently relies on historical datasets (not a live seismograph link).
*   **Data Scarcity:** Limited data for specific regions (e.g., South India).
*   **Sensor Noise:** Historical depth values in records can sometimes be approximated.

---

### 🔮 PART 9: FUTURE SCOPE
*   **Real-time Integration:** Connecting to USGS or BNS APIs for live global monitoring.
*   **Deep Learning:** Implementing **LSTM (Long Short-Term Memory)** networks for better time-series forecasting.
*   **Early Warning System:** Mobile app integration for instant proximity alerts.

---

### 🏆 PART 10: IMPACT & RELEVANCE
*   **Disaster Preparedness:** Helps identify high-probability regions for infrastructure reinforcement.
*   **Academic Value:** Demonstrates the application of Gradient Boosting on complex, real-world geospatial data.
*   **Public Awareness:** Makes complex seismic data accessible and understandable to non-technical users.

---

**Final Presentation Tip:** 
> "Judges, the most important aspect of this project is its **honesty**. While early versions showed artificial 99% accuracy due to data leakage, our current system uses a robust 26-feature pipeline that achieves a realistic R² of 0.49, making it a reliable tool for seismic pattern analysis."
