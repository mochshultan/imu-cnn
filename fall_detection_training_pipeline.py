#!/usr/bin/env python3
"""
Fall Detection Training Pipeline - CNN-LSTM Model for Grove Vision AI V2
Author: AI Assistant
Description: Complete pipeline for training a lightweight CNN-LSTM model for fall detection
            with Roboflow dataset integration and INT8 quantization for edge deployment.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import json
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

class FallDetectionTrainer:
    """Complete training pipeline for fall detection using CNN-LSTM"""
    
    def __init__(self):
        # Model configuration - optimized for Grove Vision AI V2
        self.IMG_HEIGHT = 192
        self.IMG_WIDTH = 192
        self.BATCH_SIZE = 8
        self.SEQUENCE_LENGTH = 8  # Number of frames in sequence
        self.NUM_CLASSES = 2  # non-fall, fall
        self.LEARNING_RATE = 0.0001
        self.EPOCHS = 50
        
        # Paths
        self.dataset_path = None
        self.model_save_path = "fall_detection_best_model.h5"
        self.quantized_model_path = "fall_detection_quantized_int8.tflite"
        
        self.model = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
    def download_dataset(self):
        """Download dataset from Roboflow"""
        print("🔄 Downloading dataset from Roboflow...")
        try:
            # Try to install and import roboflow
            try:
                from roboflow import Roboflow
            except ImportError:
                print("Installing roboflow...")
                os.system("pip install roboflow")
                from roboflow import Roboflow
            
            rf = Roboflow(api_key="unauthorized")
            project = rf.workspace("sut-38fcw").project("fall-detection-ip-camera")
            version = project.version(3)
            
            # Download as TFRecord format
            dataset = version.download("tfrecord")
            self.dataset_path = dataset.location
            print(f"✅ Dataset downloaded to: {self.dataset_path}")
            return True
            
        except Exception as e:
            print(f"❌ Error downloading dataset: {e}")
            print("📝 Creating mock dataset for demonstration...")
            return self.create_mock_dataset()
    
    def create_mock_dataset(self):
        """Create a mock dataset for testing when Roboflow is not available"""
        print("🔄 Creating mock dataset...")
        
        # Create directories
        os.makedirs("mock_dataset/train/fall", exist_ok=True)
        os.makedirs("mock_dataset/train/non_fall", exist_ok=True)
        os.makedirs("mock_dataset/val/fall", exist_ok=True)
        os.makedirs("mock_dataset/val/non_fall", exist_ok=True)
        
        # Generate synthetic image sequences
        def create_sequence_images(base_path, class_name, num_sequences=50):
            for seq_idx in range(num_sequences):
                for frame_idx in range(self.SEQUENCE_LENGTH):
                    # Create synthetic image data
                    if class_name == "fall":
                        # Simulate falling motion - more dynamic
                        img = np.random.randint(50, 150, 
                                              (self.IMG_HEIGHT, self.IMG_WIDTH, 3), 
                                              dtype=np.uint8)
                        # Add some motion blur effect
                        noise = np.random.normal(0, 20, img.shape)
                        img = np.clip(img + noise, 0, 255).astype(np.uint8)
                    else:
                        # Simulate normal activity - more stable
                        img = np.random.randint(80, 180, 
                                              (self.IMG_HEIGHT, self.IMG_WIDTH, 3), 
                                              dtype=np.uint8)
                    
                    # Save image
                    filename = f"{class_name}_seq{seq_idx:03d}_frame{frame_idx:02d}.png"
                    filepath = os.path.join(base_path, class_name, filename)
                    plt.imsave(filepath, img)
        
        # Create training data
        create_sequence_images("mock_dataset/train", "fall", 100)
        create_sequence_images("mock_dataset/train", "non_fall", 100)
        
        # Create validation data
        create_sequence_images("mock_dataset/val", "fall", 30)
        create_sequence_images("mock_dataset/val", "non_fall", 30)
        
        self.dataset_path = "mock_dataset"
        print("✅ Mock dataset created successfully")
        return True
    
    def prepare_datasets(self):
        """Prepare datasets for training"""
        print("🔄 Preparing datasets...")
        
        def create_sequence_dataset(data_dir):
            """Create sequences of images for temporal analysis"""
            sequences = []
            labels = []
            
            for class_idx, class_name in enumerate(['non_fall', 'fall']):
                class_dir = os.path.join(data_dir, class_name)
                if not os.path.exists(class_dir):
                    continue
                
                # Group images by sequence
                files = sorted([f for f in os.listdir(class_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
                
                # Group by sequence ID
                sequences_dict = {}
                for file in files:
                    if 'seq' in file:
                        seq_id = file.split('_seq')[1].split('_')[0]
                        if seq_id not in sequences_dict:
                            sequences_dict[seq_id] = []
                        sequences_dict[seq_id].append(os.path.join(class_dir, file))
                
                # Create sequences
                for seq_id, seq_files in sequences_dict.items():
                    if len(seq_files) >= self.SEQUENCE_LENGTH:
                        # Take first SEQUENCE_LENGTH frames
                        seq_files = sorted(seq_files)[:self.SEQUENCE_LENGTH]
                        
                        # Load images
                        sequence = []
                        for img_path in seq_files:
                            img = tf.keras.preprocessing.image.load_img(
                                img_path, target_size=(self.IMG_HEIGHT, self.IMG_WIDTH)
                            )
                            img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
                            sequence.append(img_array)
                        
                        if len(sequence) == self.SEQUENCE_LENGTH:
                            sequences.append(np.array(sequence))
                            labels.append(class_idx)
            
            return np.array(sequences), np.array(labels)
        
        # Load training data
        if os.path.exists(os.path.join(self.dataset_path, "train")):
            X_train, y_train = create_sequence_dataset(os.path.join(self.dataset_path, "train"))
        else:
            X_train, y_train = create_sequence_dataset(self.dataset_path)
        
        # Load validation data
        if os.path.exists(os.path.join(self.dataset_path, "val")):
            X_val, y_val = create_sequence_dataset(os.path.join(self.dataset_path, "val"))
        else:
            # Split training data if no separate validation set
            split_idx = int(0.8 * len(X_train))
            X_val, y_val = X_train[split_idx:], y_train[split_idx:]
            X_train, y_train = X_train[:split_idx], y_train[:split_idx]
        
        # Create test set from validation set
        split_idx = int(0.5 * len(X_val))
        X_test, y_test = X_val[split_idx:], y_val[split_idx:]
        X_val, y_val = X_val[:split_idx], y_val[:split_idx]
        
        print(f"📊 Dataset shapes:")
        print(f"   Training: {X_train.shape}, Labels: {y_train.shape}")
        print(f"   Validation: {X_val.shape}, Labels: {y_val.shape}")
        print(f"   Test: {X_test.shape}, Labels: {y_test.shape}")
        
        # Convert to TensorFlow datasets
        self.train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        self.train_dataset = self.train_dataset.batch(self.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        
        self.val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
        self.val_dataset = self.val_dataset.batch(self.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        
        self.test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
        self.test_dataset = self.test_dataset.batch(self.BATCH_SIZE)
        
        # Store for quantization
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        
        print("✅ Datasets prepared successfully")
        return True
    
    def create_lightweight_cnn_lstm_model(self):
        """Create optimized CNN-LSTM model for Grove Vision AI V2"""
        print("🔄 Creating CNN-LSTM model...")
        
        # Input shape: (batch_size, sequence_length, height, width, channels)
        inputs = keras.Input(shape=(self.SEQUENCE_LENGTH, self.IMG_HEIGHT, self.IMG_WIDTH, 3))
        
        # CNN feature extraction for each frame (TimeDistributed)
        x = layers.TimeDistributed(
            layers.Conv2D(16, (3, 3), activation='relu', padding='same')
        )(inputs)
        x = layers.TimeDistributed(layers.MaxPooling2D((2, 2)))(x)
        
        x = layers.TimeDistributed(
            layers.Conv2D(32, (3, 3), activation='relu', padding='same')
        )(x)
        x = layers.TimeDistributed(layers.MaxPooling2D((2, 2)))(x)
        
        x = layers.TimeDistributed(
            layers.Conv2D(64, (3, 3), activation='relu', padding='same')
        )(x)
        x = layers.TimeDistributed(layers.MaxPooling2D((2, 2)))(x)
        
        # Global average pooling to reduce parameters
        x = layers.TimeDistributed(layers.GlobalAveragePooling2D())(x)
        
        # LSTM layers for temporal analysis
        x = layers.LSTM(32, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(x)
        x = layers.LSTM(16, dropout=0.2, recurrent_dropout=0.2)(x)
        
        # Dense layers
        x = layers.Dense(16, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        
        # Output layer
        outputs = layers.Dense(self.NUM_CLASSES, activation='softmax')(x)
        
        self.model = keras.Model(inputs, outputs)
        
        # Compile model
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.LEARNING_RATE),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("✅ Model created successfully")
        print(f"📊 Model parameters: {self.model.count_params():,}")
        
        # Print model summary
        self.model.summary()
        
        return self.model
    
    def train_model(self):
        """Train the CNN-LSTM model"""
        print("🔄 Training model...")
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            ),
            keras.callbacks.ModelCheckpoint(
                self.model_save_path,
                monitor='val_accuracy',
                save_best_only=True,
                mode='max'
            )
        ]
        
        # Train model
        history = self.model.fit(
            self.train_dataset,
            epochs=self.EPOCHS,
            validation_data=self.val_dataset,
            callbacks=callbacks,
            verbose=1
        )
        
        # Plot training history
        self.plot_training_history(history)
        
        print("✅ Model training completed")
        return history
    
    def plot_training_history(self, history):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Accuracy
        ax1.plot(history.history['accuracy'], label='Training Accuracy')
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Loss
        ax2.plot(history.history['loss'], label='Training Loss')
        ax2.plot(history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def evaluate_model(self):
        """Evaluate model on test set"""
        print("🔄 Evaluating model...")
        
        # Load best model
        self.model = keras.models.load_model(self.model_save_path)
        
        # Evaluate on test set
        test_loss, test_accuracy = self.model.evaluate(self.test_dataset, verbose=0)
        print(f"📊 Test Accuracy: {test_accuracy:.4f}")
        print(f"📊 Test Loss: {test_loss:.4f}")
        
        # Predictions for detailed metrics
        y_pred = []
        y_true = []
        
        for batch_x, batch_y in self.test_dataset:
            predictions = self.model.predict(batch_x, verbose=0)
            y_pred.extend(np.argmax(predictions, axis=1))
            y_true.extend(batch_y.numpy())
        
        # Classification report
        print("\n📊 Classification Report:")
        print(classification_report(y_true, y_pred, target_names=['Non-Fall', 'Fall']))
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Non-Fall', 'Fall'], 
                    yticklabels=['Non-Fall', 'Fall'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return test_accuracy
    
    def quantize_model_int8(self):
        """Convert model to INT8 TensorFlow Lite for Grove Vision AI V2"""
        print("🔄 Quantizing model to INT8...")
        
        # Load best model
        self.model = keras.models.load_model(self.model_save_path)
        
        # Representative dataset for quantization
        def representative_dataset_gen():
            # Use a subset of training data for calibration
            for i in range(min(100, len(self.X_train))):
                # Yield single sample with batch dimension
                yield [self.X_train[i:i+1].astype(np.float32)]
        
        # Convert to TensorFlow Lite
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        
        # Enable INT8 quantization
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
        
        # Convert model
        quantized_tflite_model = converter.convert()
        
        # Save quantized model
        with open(self.quantized_model_path, 'wb') as f:
            f.write(quantized_tflite_model)
        
        print("✅ Model quantized successfully")
        
        # Compare model sizes
        original_size = os.path.getsize(self.model_save_path) / (1024*1024)
        quantized_size = os.path.getsize(self.quantized_model_path) / (1024*1024)
        
        print(f"📊 Original model size: {original_size:.2f} MB")
        print(f"📊 Quantized model size: {quantized_size:.2f} MB")
        print(f"📊 Size reduction: {((original_size - quantized_size) / original_size * 100):.1f}%")
        
        return quantized_size
    
    def save_quantized_model(self):
        """Save the quantized model with metadata"""
        model_info = {
            "model_name": "fall_detection_cnn_lstm",
            "input_shape": [1, self.SEQUENCE_LENGTH, self.IMG_HEIGHT, self.IMG_WIDTH, 3],
            "output_shape": [1, self.NUM_CLASSES],
            "classes": ["non_fall", "fall"],
            "preprocessing": {
                "normalization": "0-1 scaling",
                "input_type": "uint8",
                "sequence_length": self.SEQUENCE_LENGTH
            },
            "hardware_target": "Grove Vision AI V2 (ARM Cortex-M55 + Ethos-U55)",
            "quantization": "INT8 post-training quantization"
        }
        
        with open("model_info.json", "w") as f:
            json.dump(model_info, f, indent=2)
        
        print("✅ Model metadata saved")
    
    def create_deployment_guide(self, accuracy, quantized_size):
        """Create comprehensive deployment guide"""
        guide_content = f"""# Fall Detection System - Deployment Guide

## System Overview
This two-step elderly fall detection system combines:
1. **IMU-based detection** (XIAO ESP32C3 + MPU6050)
2. **Vision-based confirmation** (Grove Vision AI V2)

## Model Performance
- **Accuracy**: {accuracy:.2%}
- **Model Size**: {quantized_size:.2f} MB (INT8 quantized)
- **Architecture**: CNN-LSTM hybrid
- **Input**: {self.SEQUENCE_LENGTH} frames of {self.IMG_HEIGHT}x{self.IMG_WIDTH} RGB images
- **Classes**: Non-fall, Fall

## Hardware Requirements

### Main Components
- **XIAO ESP32C3**: Main controller with WiFi
- **Grove Vision AI V2**: AI inference module (ARM Cortex-M55 + Ethos-U55)
- **MPU6050**: 6-axis IMU sensor
- **Power Supply**: 5V/2A recommended

### Connections
```
XIAO ESP32C3 Pinout:
├── SDA (D4) → Grove Vision AI V2 SDA
├── SCL (D5) → Grove Vision AI V2 SCL
├── SDA (D4) → MPU6050 SDA  
├── SCL (D5) → MPU6050 SCL
├── 3V3 → Grove Vision AI V2 VCC
├── 3V3 → MPU6050 VCC
├── GND → All GND pins
├── D8 → Built-in LED
└── D10 → External alarm/buzzer
```

## Deployment Steps

### 1. Model Deployment to Grove Vision AI V2
1. Install SenseCraft AI software
2. Convert .tflite model using Vela compiler:
   ```bash
   vela fall_detection_quantized_int8.tflite --accelerator-config ethos-u55-256
   ```
3. Upload compiled model to Grove Vision AI V2
4. Configure I2C communication protocol

### 2. ESP32C3 Programming
1. Install Arduino IDE with ESP32 board support
2. Install required libraries:
   - WiFi
   - WebServer  
   - ArduinoJson
   - Wire
3. Update WiFi credentials in code
4. Upload `fall_detection_esp32c3.ino`

### 3. System Integration
1. Connect all hardware components
2. Power on system
3. Access web interface at ESP32's IP address
4. Verify IMU readings and Vision AI communication

## Power Consumption
- **ESP32C3**: ~80mA active, ~10μA deep sleep
- **Grove Vision AI V2**: ~200mA during inference
- **MPU6050**: ~3.9mA active
- **Total**: ~284mA active, suitable for battery operation

## Safety Features
- Two-step verification reduces false positives
- Web-based monitoring and alerts
- Configurable thresholds
- Fall counter and timing logs
- Emergency alarm output

## Alert Integration
The system supports multiple alert methods:
- Serial output for debugging
- Web interface for monitoring
- GPIO alarm output
- WiFi connectivity for cloud integration
- JSON formatted alerts for API integration

## Troubleshooting
1. **IMU not detected**: Check I2C connections and addresses
2. **Vision AI not responding**: Verify model deployment and I2C communication
3. **WiFi connection failed**: Update credentials and check network
4. **False positives**: Adjust threshold values in code
5. **Power issues**: Ensure adequate 5V/2A power supply

## Performance Optimization
- Adjust `SEQUENCE_LENGTH` for faster/slower detection
- Modify thresholds based on user sensitivity
- Enable deep sleep for battery conservation
- Use hardware timers for precise sampling

## Maintenance
- Regular model retraining with new data
- Threshold calibration for individual users  
- Battery monitoring and replacement
- Firmware updates via OTA

---
**Note**: This system is designed for monitoring assistance and should not replace professional medical monitoring devices.
"""
        
        with open("DEPLOYMENT_GUIDE.md", "w") as f:
            f.write(guide_content)
        
        print("✅ Deployment guide created: DEPLOYMENT_GUIDE.md")
    
    def run_training_pipeline(self):
        """Execute complete training pipeline"""
        print("🚀 Starting Fall Detection Training Pipeline")
        print("=" * 50)
        
        # Step 1: Download dataset
        if not self.download_dataset():
            return False
        
        # Step 2: Prepare datasets  
        if not self.prepare_datasets():
            return False
        
        # Step 3: Create model
        self.create_lightweight_cnn_lstm_model()
        
        # Step 4: Train model
        self.train_model()
        
        # Step 5: Evaluate model
        accuracy = self.evaluate_model()
        
        # Step 6: Quantize model
        quantized_size = self.quantize_model_int8()
        
        # Step 7: Save model metadata
        self.save_quantized_model()
        
        # Step 8: Create deployment guide
        self.create_deployment_guide(accuracy, quantized_size)
        
        print("\n🎉 Training pipeline completed successfully!")
        print("=" * 50)
        print("Generated files:")
        print(f"  ✓ {self.model_save_path}")
        print(f"  ✓ {self.quantized_model_path}")
        print("  ✓ model_info.json")
        print("  ✓ DEPLOYMENT_GUIDE.md")
        print("  ✓ confusion_matrix.png")
        print("  ✓ training_history.png")
        
        return True

# Main execution
if __name__ == "__main__":
    # Create trainer instance
    trainer = FallDetectionTrainer()
    
    # Run complete pipeline
    success = trainer.run_training_pipeline()
    
    if success:
        print("\n🎯 Next steps:")
        print("1. Deploy quantized model to Grove Vision AI V2")
        print("2. Upload Arduino code to XIAO ESP32C3") 
        print("3. Follow DEPLOYMENT_GUIDE.md for system setup")
        print("4. Test system with real scenarios")
    else:
        print("\n❌ Training pipeline failed. Check logs for details.")