# data-science-with-python-2024
Data Science with Python 2023 Course

# การใช้ CRISP-DM ในการพยากรณ์ราคาหุ้น SCC ปี 2024 - 2025

## 1. Business Understanding
ในโครงการนี้ เราใช้ CRISP-DM (Cross-Industry Standard Process for Data Mining) เพื่อพยากรณ์ราคาหุ้นของบริษัท SCC (Siam Cement Group) โดยใช้ข้อมูลราคาหุ้นในอดีตจากปี 2015 ถึง 2023 และพยากรณ์ราคาหุ้นในปี 2024 และ 2025 ด้วยการใช้โมเดลต่างๆ เช่น Linear Regression, Random Forest, XGBoost, และ LSTM

`วัตถุประสงค์:`
- วิเคราะห์แนวโน้มราคาหุ้น SCC
- สร้างโมเดลที่สามารถพยากรณ์ราคาหุ้นในอนาคต
- ใช้ผลลัพธ์ในการตัดสินใจลงทุน

## 2. Data Understanding
ข้อมูลที่ใช้มาจาก Yahoo Finance โดยใช้ `yfinance` เพื่อดึงข้อมูลราคาหุ้นของ SCC (ASIAN.BK) รวมถึงข้อมูลคอลัมน์สำคัญ ได้แก่:
- **Date**: `วันที่`
- **Open**: `ราคาเปิด`
- **High**: `ราคาสูงสุด`
- **Low**: `ราคาต่ำสุด`
- **Close**: `ราคาปิด`
- **Volume**: `ปริมาณการซื้อขาย`

โค้ดตัวอย่างการดึงข้อมูล:
```python
import yfinance as yf
df2 = yf.download("ASIAN.BK", start="2015-01-01", end="2023-03-11")
```

## 3. Data Preparation
3.1 **การทำความสะอาดข้อมูล**: เราตรวจสอบและลบข้อมูลที่หายไปในชุดข้อมูล
```python
print("Missing Values:\n", df2.isnull().sum())
```

3.2 **การปรับขนาดข้อมูล (Normalization)**: ใช้ `MinMaxScaler` เพื่อสเกลข้อมูลราคาหุ้นให้อยู่ในช่วง [0, 1]
```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df2[['Close']].values)
```

3.3 **การสร้างชุดข้อมูล Time Series**: สร้างข้อมูลในรูปแบบ Time Series โดยใช้ระยะเวลา 60 วันย้อนหลัง
```python
def create_dataset(dataset, time_step=60):
    X, Y = [], []
    for i in range(len(dataset) - time_step - 1):
        X.append(dataset[i:(i + time_step), 0])
        Y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(Y)
```

3.4 **การแบ่งข้อมูล Train-Test**: แบ่งข้อมูลเป็น Train Data (ข้อมูลถึงปี 2023) และ Test Data (ข้อมูลปี 2024)
```python
train = data[:'2023']
test = data['2024':]
```

## 4. Modeling and Training
ในขั้นตอนนี้เราใช้โมเดลต่างๆ เช่น Linear Regression, Random Forest, XGBoost, และ LSTM เพื่อสร้างแบบจำลองในการพยากรณ์ราคาหุ้น

โค้ดการเทรนโมเดล Linear Regression:
```python
from sklearn.linear_model import LinearRegression
lr = LinearRegression().fit(X_train, y_train)
```

## 5. การทำนายหุ้น SCC ปี 2024 และปี 2025
ใช้โมเดลที่ได้เทรนแล้วเพื่อทำนายราคาหุ้นในปี 2024 และ 2025 โดยใช้ข้อมูลถึงปี 2023 เป็นข้อมูลเริ่มต้น

โค้ดสำหรับพยากรณ์:
```python
future_dates = pd.date_range(start="2025-01-01", periods=730, freq='D')
future_pred_lr = lr.predict(future_features)
```

## 6. Evaluation
ประเมินผลความแม่นยำของแต่ละโมเดลโดยใช้ MAPE (Mean Absolute Percentage Error) เพื่อดูความแม่นยำในการพยากรณ์
```python
from sklearn.metrics import mean_absolute_percentage_error
mape_lr = mean_absolute_percentage_error(y_test, y_pred_lr) * 100
```

## 7. Deployment and Visualization of Prediction
แสดงผลการพยากรณ์ในรูปกราฟเพื่อช่วยในการตัดสินใจ เช่น การแสดงกราฟราคาหุ้นที่คาดการณ์ในอนาคต
```python
import matplotlib.pyplot as plt
plt.plot(future_dates, future_pred_lr, label="Predicted SCC Stock Price")
```

## 8. สรุปผลการวิเคราะห์
จากผลการวิเคราะห์ โมเดลต่างๆ ที่ใช้ในการพยากรณ์ราคาหุ้น SCC โดยเฉพาะ LSTM มีความแม่นยำสูงสุด และสามารถนำผลลัพธ์มาใช้เป็นข้อมูลสำหรับการตัดสินใจลงทุนในอนาคต
