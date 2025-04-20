# ASL Sensor Dataset Description

## Overview
This dataset contains sensor data for American Sign Language (ASL) alphabets, collected using wearable sensors (each with accelerometer and gyroscope). Data is organized by class (A-Z), with each class folder containing multiple Excel files representing different samples.

## Structure
- **Classes:** A-Z (26 folders), each representing an ASL alphabet.
- **Files:** Each class folder contains multiple `.xlsx` files (e.g., `A_1.xlsx`).
- **Sheets:** Each file contains multiple sheets with data from all 5 sensors.
- **Columns:**
    1. Timestamp (ignore for modeling)
    2. Sensor number (1-5)
    3. Gyroscope X (`gyro_x`)
    4. Gyroscope Y (`gyro_y`)
    5. Gyroscope Z (`gyro_z`)
    6. Accelerometer X (`acc_x`)
    7. Accelerometer Y (`acc_y`)
    8. Accelerometer Z (`acc_z`)

## Usage
- For each sample, use columns 3-8 as features.
- Each row is a time step for a specific sensor.
- Data is 3D: (time steps, features (gyro, acc), sensors (1-5))

## Example
| Timestamp | Sensor | gyro_x | gyro_y | gyro_z | acc_x | acc_y | acc_z |
|-----------|--------|--------|--------|--------|-------|-------|-------|
| 0.00      | 1      | -90    | 137    | 144    |-10932 | 9888  | 5352  |
| 0.00      | 2      | 85     | -102   | 167    | 8756  |-11245 | 4987  |

## Notes
- The 'Rest' class contains idle/non-sign data.
- Data is suitable for time-series and classification tasks.
- The 3D structure allows for analysis of sensor interactions over time.
