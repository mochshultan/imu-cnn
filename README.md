# Elderly Fall Detection using CNN-LSTM with Attention Layer

Sistem deteksi jatuh untuk lansia menggunakan model deep learning CNN-LSTM dengan attention layer yang dioptimalkan untuk deployment pada ESP32-S3.

## 📊 Dataset

Proyek ini menggunakan dataset sensor dengan struktur berikut:
- **Falls**: 2,791 samples
- **Walking**: 2,838 samples  
- **Running**: Berbagai samples
- **Stand Up**: Berbagai samples
- **Driving**: Berbagai samples

Setiap sample berisi 801 timesteps dengan 9 fitur:
- **Accelerometer**: AccX, AccY, AccZ, Magnitude
- **Gyroscope**: GyroX, GyroY, GyroZ  
- **Environment**: Temperature, Altitude

## 🧠 Model Architecture

Model menggunakan arsitektur hybrid yang menggabungkan:
- **CNN layers**: Ekstraksi pola temporal lokal
- **LSTM layers**: Menangkap dependensi jangka panjang
- **Attention mechanism**: Fokus pada timesteps penting
- **ESP32 optimization**: Kompleksitas tereduksi untuk deployment

## 🚀 Quick Start

### 1. Google Colab Setup

1. **Upload ke Google Drive:**
   - Upload folder `Dataset V4` ke Google Drive Anda
   - Pastikan struktur folder tetap sama

2. **Buka Google Colab:**
   - Upload file `elderly_fall_detection_cnn_lstm_attention.ipynb` ke Colab
   - Atau buka langsung dari GitHub

3. **Jalankan Notebook:**
   - Jalankan semua cell secara berurutan
   - Pastikan GPU runtime aktif untuk training yang lebih cepat

### 2. Output Files

Setelah training selesai, akan dihasilkan file-file berikut:
- `elderly_fall_detection_esp32s3.ino` - Arduino sketch
- `fall_detection_model.h` - Model dalam format C header
- `best_fall_detection_model.h5` - Model Keras untuk retraining
- `elderly_fall_detection_*.tflite` - Model TensorFlow Lite

## 🔧 ESP32-S3 Deployment

### Hardware Requirements

- **ESP32-S3** development board
- **MPU6050** accelerometer/gyroscope sensor
- Breadboard dan kabel jumper
- LED dan buzzer (opsional untuk alert)

### Wiring Diagram

```
ESP32-S3    MPU6050
--------    -------
3.3V    →   VCC
GND     →   GND
GPIO21  →   SDA
GPIO22  →   SCL
```

### Software Setup

1. **Install Arduino IDE** dan ESP32 board support
2. **Install Libraries:**
   ```
   - TensorFlowLite_ESP32
   - MPU6050 library (by Electronic Cats)
   - Wire library (built-in)
   ```

3. **Upload Code:**
   - Copy semua file yang dihasilkan ke folder sketch Arduino
   - Buka `elderly_fall_detection_esp32s3.ino`
   - Select board "ESP32S3 Dev Module"
   - Upload ke ESP32-S3

## 📊 Model Performance

Model yang dihasilkan memiliki performa:
- **Akurasi**: ~90-95% (tergantung dataset)
- **Precision**: ~85-92%
- **Recall**: ~88-94%
- **Model Size**: ~50-100 KB (optimized untuk ESP32)
- **Inference Time**: ~50-100ms per prediction

## 🚨 Fall Alert System

### Default Alert Mechanism
- Serial output: "*** FALL DETECTED! ***"
- LED blinking pattern (5x blink)
- Probability score logging

### Custom Alert Options
Tambahkan kode berikut di fungsi `triggerFallAlert()`:

```cpp
// WiFi notification
WiFi.begin("your_ssid", "your_password");

// Bluetooth alert
SerialBT.println("FALL_ALERT");

// Buzzer alarm
digitalWrite(BUZZER_PIN, HIGH);
delay(1000);
digitalWrite(BUZZER_PIN, LOW);
```

## 🔍 Troubleshooting

### Common Issues

1. **Memory Error pada ESP32:**
   - Kurangi `TENSOR_ARENA_SIZE`
   - Gunakan model yang lebih kecil
   - Enable PSRAM jika tersedia

2. **Low Accuracy:**
   - Tambah data training
   - Sesuaikan preprocessing parameters
   - Tuning hyperparameters

3. **False Positives:**
   - Tingkatkan fall detection threshold
   - Tambah data non-fall activities
   - Implementasi temporal filtering

## 📄 Files Generated

Setelah menjalankan notebook, Anda akan mendapatkan:

1. **elderly_fall_detection_esp32s3.ino** - Main Arduino sketch
2. **fall_detection_model.h** - TensorFlow Lite model as C header
3. **best_fall_detection_model.h5** - Keras model for future retraining
4. **elderly_fall_detection_*.tflite** - TensorFlow Lite model

## 🎛️ Customization

### Model Parameters
```python
# Untuk akurasi lebih tinggi (model lebih besar)
esp32_optimized=False

# Untuk ukuran lebih kecil (ESP32 optimized)
esp32_optimized=True
```

### Dataset Size
```python
# Full dataset
max_samples_per_class=None

# Limited dataset untuk testing cepat
max_samples_per_class=500
```

## 📚 Technical Details

### CNN-LSTM Architecture
```
Input (801, 9)
    ↓
Conv1D(16) → BatchNorm → MaxPool → Dropout
    ↓
Conv1D(32) → BatchNorm → MaxPool → Dropout
    ↓
LSTM(32) → Dropout
    ↓
LSTM(16) → Dropout
    ↓
Attention Layer
    ↓
Dense(32) → Dropout
    ↓
Dense(5) → Softmax
```

## 🔄 Future Improvements

1. **Data Augmentation**: Synthetic data generation
2. **Transfer Learning**: Pretrained model adaptation
3. **Real-time Streaming**: Continuous monitoring
4. **Multi-sensor Fusion**: Tambah sensor lain (heart rate, GPS)
5. **Edge AI Optimization**: Lebih lanjut optimize untuk edge devices

---

**⚠️ Disclaimer**: Sistem ini untuk tujuan edukasi dan penelitian. Untuk penggunaan medis atau keselamatan kritis, diperlukan validasi dan sertifikasi tambahan.
