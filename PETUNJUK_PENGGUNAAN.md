# Petunjuk Penggunaan Sistem Deteksi Jatuh Lansia

## 🎯 Tujuan Proyek
Membuat sistem deteksi jatuh otomatis untuk lansia menggunakan:
- Model CNN-LSTM dengan Attention Layer
- Sensor accelerometer dan gyroscope (MPU6050)
- Deployment pada ESP32-S3

## 📋 Langkah-langkah Penggunaan

### 1. Persiapan Dataset
```
✅ Dataset sudah tersedia di folder "Dataset V4"
✅ Berisi 5 kategori aktivitas: Falls, Walking, Running, Stand Up, Driving
✅ Total: 8,956 file CSV dengan 801 timesteps per file
```

### 2. Training Model di Google Colab

#### A. Upload ke Google Drive
1. Buka Google Drive
2. Buat folder baru (misal: "Fall Detection Project")
3. Upload folder "Dataset V4" ke dalam folder tersebut
4. Upload file "elderly_fall_detection_cnn_lstm_attention.ipynb"

#### B. Jalankan Training
1. Buka notebook di Google Colab
2. Pastikan Runtime type = GPU (Runtime → Change runtime type → GPU)
3. Jalankan semua cell secara berurutan:
   - Cell 1: Install packages
   - Cell 2: Mount Google Drive
   - Cell 3-4: Load dataset
   - Cell 5-6: Preprocessing
   - Cell 7-8: Model architecture
   - Cell 9: Training (akan memakan waktu 20-60 menit)
   - Cell 10: Evaluation
   - Cell 11-12: Konversi ke TensorFlow Lite
   - Cell 13-14: Generate Arduino code

#### C. Download Hasil Training
Setelah training selesai, download file-file berikut:
- `elderly_fall_detection_esp32s3.ino`
- `fall_detection_model.h`
- `best_fall_detection_model.h5`
- `elderly_fall_detection_*.tflite`

### 3. Setup Hardware ESP32-S3

#### A. Komponen yang Diperlukan
- ESP32-S3 development board
- Sensor MPU6050
- Kabel jumper
- Breadboard
- LED dan resistor (opsional)
- Buzzer (opsional)

#### B. Wiring Connections
```
ESP32-S3 Pin    MPU6050 Pin    Keterangan
============    ===========    ==========
3.3V            VCC            Power supply
GND             GND            Ground
GPIO21          SDA            I2C Data
GPIO22          SCL            I2C Clock
```

### 4. Setup Software Arduino

#### A. Install Arduino IDE
1. Download Arduino IDE terbaru
2. Install ESP32 board package:
   - File → Preferences
   - Additional Board Manager URLs: 
     `https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json`
   - Tools → Board → Boards Manager → Search "ESP32" → Install

#### B. Install Libraries
Buka Library Manager (Tools → Manage Libraries) dan install:
- `TensorFlowLite_ESP32` by Eloquent Arduino
- `MPU6050` by Electronic Cats
- `Wire` (sudah built-in)

#### C. Upload Code
1. Copy semua file hasil training ke folder sketch Arduino
2. Buka `elderly_fall_detection_esp32s3.ino`
3. Select Board: "ESP32S3 Dev Module"
4. Select Port yang sesuai
5. Click Upload

### 5. Testing dan Kalibrasi

#### A. Serial Monitor
1. Buka Serial Monitor (Tools → Serial Monitor)
2. Set baud rate ke 115200
3. Observe output untuk memastikan:
   - MPU6050 terhubung dengan benar
   - Model loaded successfully
   - Sensor readings normal

#### B. Testing Gerakan
Lakukan berbagai gerakan dan amati output:
- **Walking**: Harus terdeteksi sebagai "Walking"
- **Standing**: Harus terdeteksi sebagai "Stand Up"
- **Simulated Fall**: Jatuhkan device dengan aman

#### C. Adjust Threshold
Jika terlalu banyak false positive/negative, edit di Arduino code:
```cpp
// Ubah nilai threshold (0.0 - 1.0)
if (strcmp(ACTIVITY_LABELS[predicted_class], "Falls") == 0 && max_probability > 0.7) {
    // Ubah 0.7 ke nilai yang sesuai
```

### 6. Customization Alert System

#### A. LED Alert
```cpp
void triggerFallAlert() {
    // LED blink pattern
    for (int i = 0; i < 10; i++) {
        digitalWrite(LED_PIN, HIGH);
        delay(100);
        digitalWrite(LED_PIN, LOW);
        delay(100);
    }
}
```

#### B. Buzzer Alert
```cpp
void triggerFallAlert() {
    // Buzzer alarm
    digitalWrite(BUZZER_PIN, HIGH);
    delay(2000);
    digitalWrite(BUZZER_PIN, LOW);
}
```

#### C. WiFi Notification
```cpp
#include <WiFi.h>
#include <HTTPClient.h>

void triggerFallAlert() {
    HTTPClient http;
    http.begin("http://your-server.com/fall-alert");
    http.addHeader("Content-Type", "application/json");
    
    String payload = "{\"alert\":\"fall_detected\",\"timestamp\":\"" + String(millis()) + "\"}";
    int httpResponseCode = http.POST(payload);
    
    http.end();
}
```

## 🔧 Troubleshooting

### Masalah Umum dan Solusi

#### 1. Model Tidak Load
**Error**: "Model schema version mismatch"
**Solusi**: 
- Pastikan menggunakan TensorFlow Lite versi yang kompatibel
- Regenerate model header file

#### 2. Memory Error
**Error**: "AllocateTensors() failed"
**Solusi**:
- Kurangi `TENSOR_ARENA_SIZE` dari 70000 ke 50000
- Enable PSRAM di Arduino IDE (Tools → PSRAM → Enabled)

#### 3. Sensor Tidak Terdeteksi
**Error**: "MPU6050 connection failed"
**Solusi**:
- Periksa wiring connections
- Pastikan MPU6050 mendapat power 3.3V
- Coba ganti address I2C (0x68 atau 0x69)

#### 4. Akurasi Rendah
**Gejala**: Banyak false positive/negative
**Solusi**:
- Kalibrasi sensor di posisi datar
- Adjust detection threshold
- Tambah data training dengan kondisi berbeda

### Debug Tips

```cpp
// Enable debug mode
#define DEBUG_MODE 1

#if DEBUG_MODE
  Serial.printf("Sensor: %.3f,%.3f,%.3f\n", acc_x, acc_y, acc_z);
  Serial.printf("Prediction: %s (%.2f%%)\n", predicted_class, probability);
#endif
```

## 📊 Expected Performance

### Model Metrics
- **Akurasi**: 85-95%
- **Precision**: 80-92%
- **Recall**: 85-94%
- **Model Size**: 50-100 KB
- **Inference Time**: 50-100ms

### Real-world Performance
- **Detection Delay**: 2-5 detik setelah kejadian
- **False Positive Rate**: <5% dengan threshold optimal
- **Battery Life**: 8-12 jam (tergantung implementasi)

## 🎯 Rekomendasi Deployment

### Untuk Penelitian
- Gunakan model full (esp32_optimized=False)
- Log semua predictions untuk analisis
- Implement data collection untuk improvement

### Untuk Produksi
- Gunakan model optimized (esp32_optimized=True)
- Implement battery optimization
- Add wireless connectivity untuk emergency alert
- Tambah GPS tracking

## 📞 Support

Jika mengalami masalah:
1. Periksa wiring dan connections
2. Pastikan semua libraries ter-install dengan benar
3. Check Serial Monitor untuk error messages
4. Bandingkan dengan kode template yang provided

---

**✅ Checklist Sukses:**
- [ ] Model berhasil di-training di Colab
- [ ] Files Arduino ter-generate dengan benar
- [ ] Hardware terhubung dan sensor berfungsi
- [ ] Model berhasil di-upload ke ESP32-S3
- [ ] Fall detection berjalan dengan akurasi yang baik
- [ ] Alert system berfungsi dengan baik

**🎉 Selamat! Sistem deteksi jatuh lansia Anda sudah siap digunakan!**
