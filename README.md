
# ♻️ Smart Recycling Bin with Machine Learning and Arduino

## Overview
This project is a **smart recycling bin** that only opens when it detects a recyclable item.  
It combines **machine learning**, **Arduino-controlled motors**, and **Python code** to utilize **mechanics, electronics, and programming**.

---

## 🎯 Purpose of the Project

The goal of this project is to reduce contamination in recycling streams by ensuring that only recyclable items enter the bin.  
Contamination occurs when **non-recyclable items** are thrown into recycling bins, causing entire batches to be rejected and sent to landfills. 
  

It was developed as a personal learning project to explore robotics, ML classification, and hardware-software integration, and to create a working prototype that addresses a real environmental problem.

---

## ✨ Features
- Detects recyclable items using a machine learning model 
- Opens a **cardboard lid via Arduino-controlled motors** when the item is recognized  
- Prevents **non-recyclables** from entering the bin  
- **demonstration video available on YouTube**

---

## ⚙️ Hardware
- Arduino UNO
- AdaFruit Motor Shield
- DC Motors to open the lid  
- Cardboard box (as the bin)  
- USB Cable  

---

## 💻 Software
- **Python scripts**   
- **Arduino code** for motor control  
- **Picture Database** 

---

## 🧠 Installation / Usage
1. Connect **Arduino** to your computer via USB.  
2. Upload the **Arduino code** to the board.  
3. Run the **Python script** to start the detection system.  
4. Place recyclable items in front of the sensor/camera and the lid will **open automatically** when detected.

---

## 🎥 Project Video
Watch the full demonstration on YouTube:  
👉 [Smart Recycling Bin Demonstration](https://youtu.be/0VyxrFK3I_0)

---

## 📝 Notes
- Developed as a **personal robotics and machine learning project**  
- Trained only on a limited database, can be expanded easily though.

---

## 🚀 Implemented Improvements

- Added **LED indicators** to provide real-time status, for improved usability:
  - **Red LED** stays ON during standby.
  - **Green LED** turns ON during correct item detection and lid opening.

---

