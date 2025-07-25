import json

def create_complete_notebook():
    cells = []
    
    # Title cell
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Elderly Fall Detection using CNN-LSTM with Attention Layer\n\n"
            "This notebook trains a deep learning model for detecting elderly falls using accelerometer and gyroscope data. "
            "The trained model will be optimized for deployment on ESP32-S3 microcontroller.\n\n"
            "## Dataset Structure\n"
            "- **Falls**: 2,791 samples\n"
            "- **Walking**: 2,838 samples\n"
            "- **Running**: Various samples\n"
            "- **Stand Up**: Various samples\n"
            "- **Driving**: Various samples\n\n"
            "Each sample contains 801 time steps with 9 features:\n"
            "- AccX, AccY, AccZ (accelerometer)\n"
            "- GyroX, GyroY, GyroZ (gyroscope)\n"
            "- Magnitude, Temperature, Altitude"
        ]
    })
    
    # Setup section
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": "## 1. Setup and Installation\n\nFirst, let's install required packages and setup the environment for Google Colab."
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Install required packages\n"
            "!pip install tensorflow==2.13.0\n"
            "!pip install numpy pandas matplotlib scikit-learn seaborn\n"
            "!pip install tensorflow-model-optimization\n\n"
            "# For ESP32 deployment\n"
            "!pip install tensorflow-lite-runtime\n\n"
            "import os\n"
            "import numpy as np\n"
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n"
            "import seaborn as sns\n"
            "from sklearn.preprocessing import StandardScaler, LabelEncoder\n"
            "from sklearn.model_selection import train_test_split\n"
            "from sklearn.metrics import classification_report, confusion_matrix, accuracy_score\n"
            "import tensorflow as tf\n"
            "from tensorflow import keras\n"
            "from tensorflow.keras import layers, models, optimizers, callbacks\n"
            "from tensorflow.keras.utils import to_categorical\n"
            "import glob\n"
            "import warnings\n"
            "warnings.filterwarnings('ignore')\n\n"
            "print(f\"TensorFlow version: {tf.__version__}\")\n"
            "print(f\"GPU Available: {tf.test.is_gpu_available()}\")\n"
            "print(f\"GPU Devices: {tf.config.list_physical_devices('GPU')}\")"
        ]
    })
    
    # Drive mount section
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": "## 2. Mount Google Drive and Upload Dataset\n\nUpload your 'Dataset V4' folder to Google Drive and mount it here."
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from google.colab import drive\n"
            "drive.mount('/content/drive')\n\n"
            "# Update this path to where you uploaded your dataset in Google Drive\n"
            "DATASET_PATH = '/content/drive/MyDrive/Dataset V4'\n\n"
            "# Verify dataset exists\n"
            "if os.path.exists(DATASET_PATH):\n"
            "    print(\"Dataset found!\")\n"
            "    categories = os.listdir(DATASET_PATH)\n"
            "    print(f\"Categories: {categories}\")\n"
            "    \n"
            "    for category in categories:\n"
            "        if os.path.isdir(os.path.join(DATASET_PATH, category)):\n"
            "            count = len(glob.glob(os.path.join(DATASET_PATH, category, '*.csv')))\n"
            "            print(f\"{category}: {count} files\")\n"
            "else:\n"
            "    print(\"Dataset not found. Please upload 'Dataset V4' to your Google Drive.\")"
        ]
    })
    
    # Data loading section
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": "## 3. Data Loading and Preprocessing"
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "def load_sensor_data(dataset_path, max_samples_per_class=None, sequence_length=801):\n"
            "    \"\"\"\n"
            "    Load and preprocess sensor data from CSV files\n"
            "    \n"
            "    Args:\n"
            "        dataset_path: Path to dataset directory\n"
            "        max_samples_per_class: Maximum samples per activity class (None for all)\n"
            "        sequence_length: Number of time steps per sequence\n"
            "    \n"
            "    Returns:\n"
            "        X: Feature data (samples, time_steps, features)\n"
            "        y: Labels\n"
            "        label_encoder: Fitted label encoder\n"
            "    \"\"\"\n"
            "    \n"
            "    X = []\n"
            "    y = []\n"
            "    \n"
            "    # Define activity categories\n"
            "    categories = ['Falls', 'Walking', 'Running', 'Stand Up', 'Driving']\n"
            "    \n"
            "    print(\"Loading data...\")\n"
            "    \n"
            "    for category in categories:\n"
            "        category_path = os.path.join(dataset_path, category)\n"
            "        if not os.path.exists(category_path):\n"
            "            print(f\"Warning: {category} folder not found\")\n"
            "            continue\n"
            "            \n"
            "        csv_files = glob.glob(os.path.join(category_path, '*.csv'))\n"
            "        \n"
            "        if max_samples_per_class:\n"
            "            csv_files = csv_files[:max_samples_per_class]\n"
            "        \n"
            "        print(f\"Processing {category}: {len(csv_files)} files\")\n"
            "        \n"
            "        for i, file_path in enumerate(csv_files):\n"
            "            try:\n"
            "                # Read CSV file\n"
            "                df = pd.read_csv(file_path)\n"
            "                \n"
            "                # Select relevant sensor features\n"
            "                feature_columns = ['AccX', 'AccY', 'AccZ', 'Magnitude', \n"
            "                                 'GyroX', 'GyroY', 'GyroZ', 'Temperature', 'Altitude']\n"
            "                \n"
            "                # Ensure all columns exist\n"
            "                available_columns = [col for col in feature_columns if col in df.columns]\n"
            "                \n"
            "                if len(available_columns) < 6:  # At least acc and gyro data\n"
            "                    print(f\"Warning: Insufficient features in {file_path}\")\n"
            "                    continue\n"
            "                \n"
            "                # Extract features\n"
            "                features = df[available_columns].values\n"
            "                \n"
            "                # Pad or truncate to fixed sequence length\n"
            "                if len(features) > sequence_length:\n"
            "                    features = features[:sequence_length]\n"
            "                elif len(features) < sequence_length:\n"
            "                    padding = np.zeros((sequence_length - len(features), len(available_columns)))\n"
            "                    features = np.vstack([features, padding])\n"
            "                \n"
            "                # Handle NaN values\n"
            "                features = np.nan_to_num(features, nan=0.0)\n"
            "                \n"
            "                X.append(features)\n"
            "                y.append(category)\n"
            "                \n"
            "                if (i + 1) % 100 == 0:\n"
            "                    print(f\"  Processed {i + 1}/{len(csv_files)} files\")\n"
            "                    \n"
            "            except Exception as e:\n"
            "                print(f\"Error processing {file_path}: {str(e)}\")\n"
            "                continue\n"
            "    \n"
            "    # Convert to numpy arrays\n"
            "    X = np.array(X)\n"
            "    y = np.array(y)\n"
            "    \n"
            "    # Encode labels\n"
            "    label_encoder = LabelEncoder()\n"
            "    y_encoded = label_encoder.fit_transform(y)\n"
            "    \n"
            "    print(f\"\\nData loading completed:\")\n"
            "    print(f\"Total samples: {len(X)}\")\n"
            "    print(f\"Input shape: {X.shape}\")\n"
            "    print(f\"Classes: {label_encoder.classes_}\")\n"
            "    print(f\"Class distribution:\")\n"
            "    unique, counts = np.unique(y, return_counts=True)\n"
            "    for cls, count in zip(unique, counts):\n"
            "        print(f\"  {cls}: {count} samples\")\n"
            "    \n"
            "    return X, y_encoded, label_encoder"
        ]
    })
    
    # Load dataset cell
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Load the dataset\n"
            "# Reduce max_samples_per_class for faster training during development\n"
            "X, y, label_encoder = load_sensor_data(\n"
            "    DATASET_PATH, \n"
            "    max_samples_per_class=500,  # Increase this or set to None for full dataset\n"
            "    sequence_length=801\n"
            ")\n\n"
            "print(f\"\\nFinal dataset shape: {X.shape}\")\n"
            "print(f\"Number of classes: {len(label_encoder.classes_)}\")\n"
            "print(f\"Features per timestep: {X.shape[2]}\")"
        ]
    })
    
    # Preprocessing section
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": "## 4. Data Preprocessing and Normalization"
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Normalize the data\n"
            "def normalize_data(X_train, X_val, X_test):\n"
            "    \"\"\"\n"
            "    Normalize the sensor data using StandardScaler\n"
            "    \"\"\"\n"
            "    # Reshape for scaling (samples * timesteps, features)\n"
            "    original_shape = X_train.shape\n"
            "    X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])\n"
            "    X_val_reshaped = X_val.reshape(-1, X_val.shape[-1])\n"
            "    X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])\n"
            "    \n"
            "    # Fit scaler on training data\n"
            "    scaler = StandardScaler()\n"
            "    X_train_scaled = scaler.fit_transform(X_train_reshaped)\n"
            "    X_val_scaled = scaler.transform(X_val_reshaped)\n"
            "    X_test_scaled = scaler.transform(X_test_reshaped)\n"
            "    \n"
            "    # Reshape back to original shape\n"
            "    X_train_scaled = X_train_scaled.reshape(original_shape)\n"
            "    X_val_scaled = X_val_scaled.reshape(X_val.shape)\n"
            "    X_test_scaled = X_test_scaled.reshape(X_test.shape)\n"
            "    \n"
            "    return X_train_scaled, X_val_scaled, X_test_scaled, scaler\n\n"
            "# Split the data\n"
            "X_temp, X_test, y_temp, y_test = train_test_split(\n"
            "    X, y, test_size=0.2, random_state=42, stratify=y\n"
            ")\n\n"
            "X_train, X_val, y_train, y_val = train_test_split(\n"
            "    X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp\n"
            ")\n\n"
            "print(f\"Training set: {X_train.shape}\")\n"
            "print(f\"Validation set: {X_val.shape}\")\n"
            "print(f\"Test set: {X_test.shape}\")\n\n"
            "# Normalize the data\n"
            "X_train_scaled, X_val_scaled, X_test_scaled, scaler = normalize_data(X_train, X_val, X_test)\n\n"
            "# Convert labels to categorical\n"
            "num_classes = len(label_encoder.classes_)\n"
            "y_train_cat = to_categorical(y_train, num_classes)\n"
            "y_val_cat = to_categorical(y_val, num_classes)\n"
            "y_test_cat = to_categorical(y_test, num_classes)\n\n"
            "print(f\"\\nNormalized data shapes:\")\n"
            "print(f\"X_train: {X_train_scaled.shape}\")\n"
            "print(f\"y_train: {y_train_cat.shape}\")\n"
            "print(f\"Number of classes: {num_classes}\")"
        ]
    })
    
    # Model architecture section
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 5. CNN-LSTM Model with Attention Layer\n\n"
            "We'll create a hybrid model combining:\n"
            "- **CNN layers**: For extracting local temporal patterns\n"
            "- **LSTM layers**: For capturing long-term dependencies\n"
            "- **Attention mechanism**: For focusing on important time steps\n"
            "- **Optimization for ESP32**: Reduced complexity for deployment"
        ]
    })
    
    # Attention layer definition
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Custom Attention Layer\n"
            "class AttentionLayer(layers.Layer):\n"
            "    def __init__(self, **kwargs):\n"
            "        super(AttentionLayer, self).__init__(**kwargs)\n"
            "        \n"
            "    def build(self, input_shape):\n"
            "        self.W = self.add_weight(\n"
            "            name='attention_weight',\n"
            "            shape=(input_shape[-1], 1),\n"
            "            initializer='random_normal',\n"
            "            trainable=True\n"
            "        )\n"
            "        self.b = self.add_weight(\n"
            "            name='attention_bias',\n"
            "            shape=(input_shape[1], 1),\n"
            "            initializer='zeros',\n"
            "            trainable=True\n"
            "        )\n"
            "        super(AttentionLayer, self).build(input_shape)\n"
            "        \n"
            "    def call(self, inputs):\n"
            "        # inputs shape: (batch_size, time_steps, features)\n"
            "        # Compute attention scores\n"
            "        attention_scores = tf.nn.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)\n"
            "        attention_weights = tf.nn.softmax(attention_scores, axis=1)\n"
            "        \n"
            "        # Apply attention weights\n"
            "        weighted_input = inputs * attention_weights\n"
            "        output = tf.reduce_sum(weighted_input, axis=1)\n"
            "        \n"
            "        return output\n"
            "    \n"
            "    def get_config(self):\n"
            "        return super(AttentionLayer, self).get_config()"
        ]
    })
    
    # Model creation
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "def create_cnn_lstm_attention_model(input_shape, num_classes, esp32_optimized=True):\n"
            "    \"\"\"\n"
            "    Create CNN-LSTM model with attention layer\n"
            "    \n"
            "    Args:\n"
            "        input_shape: Shape of input data (time_steps, features)\n"
            "        num_classes: Number of output classes\n"
            "        esp32_optimized: If True, use smaller model for ESP32 deployment\n"
            "    \"\"\"\n"
            "    \n"
            "    model = models.Sequential()\n"
            "    \n"
            "    # Input layer\n"
            "    model.add(layers.Input(shape=input_shape))\n"
            "    \n"
            "    if esp32_optimized:\n"
            "        # Optimized for ESP32 - smaller model\n"
            "        # CNN layers for feature extraction\n"
            "        model.add(layers.Conv1D(filters=16, kernel_size=3, activation='relu', padding='same'))\n"
            "        model.add(layers.BatchNormalization())\n"
            "        model.add(layers.MaxPooling1D(pool_size=2))\n"
            "        model.add(layers.Dropout(0.3))\n"
            "        \n"
            "        model.add(layers.Conv1D(filters=32, kernel_size=3, activation='relu', padding='same'))\n"
            "        model.add(layers.BatchNormalization())\n"
            "        model.add(layers.MaxPooling1D(pool_size=2))\n"
            "        model.add(layers.Dropout(0.3))\n"
            "        \n"
            "        # LSTM layers\n"
            "        model.add(layers.LSTM(32, return_sequences=True, dropout=0.3, recurrent_dropout=0.2))\n"
            "        model.add(layers.LSTM(16, return_sequences=True, dropout=0.3, recurrent_dropout=0.2))\n"
            "        \n"
            "        # Attention layer\n"
            "        model.add(AttentionLayer())\n"
            "        \n"
            "        # Dense layers\n"
            "        model.add(layers.Dense(32, activation='relu'))\n"
            "        model.add(layers.Dropout(0.5))\n"
            "        \n"
            "    else:\n"
            "        # Full model for better accuracy\n"
            "        # CNN layers\n"
            "        model.add(layers.Conv1D(filters=32, kernel_size=5, activation='relu', padding='same'))\n"
            "        model.add(layers.BatchNormalization())\n"
            "        model.add(layers.MaxPooling1D(pool_size=2))\n"
            "        model.add(layers.Dropout(0.3))\n"
            "        \n"
            "        model.add(layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'))\n"
            "        model.add(layers.BatchNormalization())\n"
            "        model.add(layers.MaxPooling1D(pool_size=2))\n"
            "        model.add(layers.Dropout(0.3))\n"
            "        \n"
            "        model.add(layers.Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'))\n"
            "        model.add(layers.BatchNormalization())\n"
            "        model.add(layers.MaxPooling1D(pool_size=2))\n"
            "        model.add(layers.Dropout(0.3))\n"
            "        \n"
            "        # LSTM layers\n"
            "        model.add(layers.LSTM(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.2))\n"
            "        model.add(layers.LSTM(32, return_sequences=True, dropout=0.3, recurrent_dropout=0.2))\n"
            "        \n"
            "        # Attention layer\n"
            "        model.add(AttentionLayer())\n"
            "        \n"
            "        # Dense layers\n"
            "        model.add(layers.Dense(64, activation='relu'))\n"
            "        model.add(layers.Dropout(0.5))\n"
            "        model.add(layers.Dense(32, activation='relu'))\n"
            "        model.add(layers.Dropout(0.5))\n"
            "    \n"
            "    # Output layer\n"
            "    model.add(layers.Dense(num_classes, activation='softmax'))\n"
            "    \n"
            "    return model\n\n"
            "# Create the model\n"
            "input_shape = (X_train_scaled.shape[1], X_train_scaled.shape[2])\n"
            "print(f\"Input shape: {input_shape}\")\n\n"
            "# Create ESP32-optimized model\n"
            "model = create_cnn_lstm_attention_model(input_shape, num_classes, esp32_optimized=True)\n\n"
            "# Compile the model\n"
            "model.compile(\n"
            "    optimizer=optimizers.Adam(learning_rate=0.001),\n"
            "    loss='categorical_crossentropy',\n"
            "    metrics=['accuracy', 'precision', 'recall']\n"
            ")\n\n"
            "# Print model summary\n"
            "model.summary()\n\n"
            "# Calculate model size\n"
            "param_count = model.count_params()\n"
            "print(f\"\\nTotal parameters: {param_count:,}\")\n"
            "print(f\"Estimated model size: ~{param_count * 4 / 1024 / 1024:.2f} MB (32-bit floats)\")"
        ]
    })
    
    # Training section
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": "## 6. Model Training"
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Create callbacks for training\n"
            "early_stopping = callbacks.EarlyStopping(\n"
            "    monitor='val_loss',\n"
            "    patience=10,\n"
            "    restore_best_weights=True\n"
            ")\n\n"
            "reduce_lr = callbacks.ReduceLROnPlateau(\n"
            "    monitor='val_loss',\n"
            "    factor=0.5,\n"
            "    patience=5,\n"
            "    min_lr=1e-7\n"
            ")\n\n"
            "model_checkpoint = callbacks.ModelCheckpoint(\n"
            "    'best_fall_detection_model.h5',\n"
            "    monitor='val_accuracy',\n"
            "    save_best_only=True,\n"
            "    mode='max'\n"
            ")\n\n"
            "callbacks_list = [early_stopping, reduce_lr, model_checkpoint]\n\n"
            "# Train the model\n"
            "print(\"Starting model training...\")\n"
            "print(f\"Training samples: {len(X_train_scaled)}\")\n"
            "print(f\"Validation samples: {len(X_val_scaled)}\")\n\n"
            "# Training parameters\n"
            "EPOCHS = 50\n"
            "BATCH_SIZE = 32\n\n"
            "# Start training\n"
            "history = model.fit(\n"
            "    X_train_scaled, y_train_cat,\n"
            "    batch_size=BATCH_SIZE,\n"
            "    epochs=EPOCHS,\n"
            "    validation_data=(X_val_scaled, y_val_cat),\n"
            "    callbacks=callbacks_list,\n"
            "    verbose=1\n"
            ")\n\n"
            "print(\"\\nTraining completed!\")"
        ]
    })
    
    # Evaluation section
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": "## 7. Model Evaluation and Visualization"
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Plot training history\n"
            "def plot_training_history(history):\n"
            "    fig, axes = plt.subplots(2, 2, figsize=(15, 10))\n"
            "    \n"
            "    # Accuracy\n"
            "    axes[0, 0].plot(history.history['accuracy'], label='Training Accuracy')\n"
            "    axes[0, 0].plot(history.history['val_accuracy'], label='Validation Accuracy')\n"
            "    axes[0, 0].set_title('Model Accuracy')\n"
            "    axes[0, 0].set_xlabel('Epoch')\n"
            "    axes[0, 0].set_ylabel('Accuracy')\n"
            "    axes[0, 0].legend()\n"
            "    axes[0, 0].grid(True)\n"
            "    \n"
            "    # Loss\n"
            "    axes[0, 1].plot(history.history['loss'], label='Training Loss')\n"
            "    axes[0, 1].plot(history.history['val_loss'], label='Validation Loss')\n"
            "    axes[0, 1].set_title('Model Loss')\n"
            "    axes[0, 1].set_xlabel('Epoch')\n"
            "    axes[0, 1].set_ylabel('Loss')\n"
            "    axes[0, 1].legend()\n"
            "    axes[0, 1].grid(True)\n"
            "    \n"
            "    # Precision\n"
            "    if 'precision' in history.history:\n"
            "        axes[1, 0].plot(history.history['precision'], label='Training Precision')\n"
            "        axes[1, 0].plot(history.history['val_precision'], label='Validation Precision')\n"
            "        axes[1, 0].set_title('Model Precision')\n"
            "        axes[1, 0].set_xlabel('Epoch')\n"
            "        axes[1, 0].set_ylabel('Precision')\n"
            "        axes[1, 0].legend()\n"
            "        axes[1, 0].grid(True)\n"
            "    \n"
            "    # Recall\n"
            "    if 'recall' in history.history:\n"
            "        axes[1, 1].plot(history.history['recall'], label='Training Recall')\n"
            "        axes[1, 1].plot(history.history['val_recall'], label='Validation Recall')\n"
            "        axes[1, 1].set_title('Model Recall')\n"
            "        axes[1, 1].set_xlabel('Epoch')\n"
            "        axes[1, 1].set_ylabel('Recall')\n"
            "        axes[1, 1].legend()\n"
            "        axes[1, 1].grid(True)\n"
            "    \n"
            "    plt.tight_layout()\n"
            "    plt.show()\n\n"
            "plot_training_history(history)\n\n"
            "# Load the best model and evaluate on test set\n"
            "model.load_weights('best_fall_detection_model.h5')\n\n"
            "test_loss, test_accuracy, test_precision, test_recall = model.evaluate(\n"
            "    X_test_scaled, y_test_cat, verbose=0\n"
            ")\n\n"
            "print(f\"Test Results:\")\n"
            "print(f\"Test Accuracy: {test_accuracy:.4f}\")\n"
            "print(f\"Test Precision: {test_precision:.4f}\")\n"
            "print(f\"Test Recall: {test_recall:.4f}\")\n"
            "print(f\"Test Loss: {test_loss:.4f}\")\n\n"
            "# Generate predictions\n"
            "y_pred = model.predict(X_test_scaled)\n"
            "y_pred_classes = np.argmax(y_pred, axis=1)\n"
            "y_true_classes = np.argmax(y_test_cat, axis=1)\n\n"
            "# Classification report\n"
            "print(\"\\nClassification Report:\")\n"
            "print(classification_report(\n"
            "    y_true_classes, y_pred_classes, \n"
            "    target_names=label_encoder.classes_\n"
            "))\n\n"
            "# Confusion Matrix\n"
            "cm = confusion_matrix(y_true_classes, y_pred_classes)\n"
            "plt.figure(figsize=(10, 8))\n"
            "sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',\n"
            "            xticklabels=label_encoder.classes_,\n"
            "            yticklabels=label_encoder.classes_)\n"
            "plt.title('Confusion Matrix')\n"
            "plt.xlabel('Predicted')\n"
            "plt.ylabel('Actual')\n"
            "plt.show()"
        ]
    })
    
    # Model conversion section
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 8. Model Conversion for ESP32-S3\n\n"
            "Convert the trained model to TensorFlow Lite format for deployment on ESP32-S3."
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Convert model to TensorFlow Lite\n"
            "def convert_to_tflite(model, model_name=\"fall_detection_model\"):\n"
            "    \"\"\"\n"
            "    Convert Keras model to TensorFlow Lite format\n"
            "    \"\"\"\n"
            "    # Basic conversion\n"
            "    converter = tf.lite.TFLiteConverter.from_keras_model(model)\n"
            "    \n"
            "    # Apply optimizations for microcontroller deployment\n"
            "    converter.optimizations = [tf.lite.Optimize.DEFAULT]\n"
            "    \n"
            "    # Try quantization for smaller size\n"
            "    try:\n"
            "        # Representative dataset for quantization\n"
            "        def representative_dataset():\n"
            "            for i in range(min(100, len(X_train_scaled))):\n"
            "                data = X_train_scaled[i:i+1].astype(np.float32)\n"
            "                yield [data]\n"
            "        \n"
            "        converter.representative_dataset = representative_dataset\n"
            "        converter.target_spec.supported_types = [tf.int8]\n"
            "        converter.inference_input_type = tf.int8\n"
            "        converter.inference_output_type = tf.int8\n"
            "        \n"
            "        tflite_model = converter.convert()\n"
            "        filename = f'{model_name}_quantized.tflite'\n"
            "        \n"
            "    except Exception as e:\n"
            "        print(f\"Quantized conversion failed: {e}\")\n"
            "        print(\"Using basic conversion...\")\n"
            "        \n"
            "        # Fallback to basic conversion\n"
            "        converter = tf.lite.TFLiteConverter.from_keras_model(model)\n"
            "        converter.optimizations = [tf.lite.Optimize.DEFAULT]\n"
            "        \n"
            "        tflite_model = converter.convert()\n"
            "        filename = f'{model_name}_basic.tflite'\n"
            "    \n"
            "    # Save the model\n"
            "    with open(filename, 'wb') as f:\n"
            "        f.write(tflite_model)\n"
            "    \n"
            "    print(f\"TFLite model saved as {filename}\")\n"
            "    print(f\"Model size: {len(tflite_model) / 1024:.2f} KB\")\n"
            "    \n"
            "    return tflite_model, filename\n\n"
            "# Convert the model\n"
            "tflite_model, tflite_filename = convert_to_tflite(model, \"elderly_fall_detection\")\n\n"
            "# Test TFLite model\n"
            "interpreter = tf.lite.Interpreter(model_content=tflite_model)\n"
            "interpreter.allocate_tensors()\n\n"
            "# Get input and output details\n"
            "input_details = interpreter.get_input_details()\n"
            "output_details = interpreter.get_output_details()\n\n"
            "print(f\"\\nTFLite Model Details:\")\n"
            "print(f\"Input shape: {input_details[0]['shape']}\")\n"
            "print(f\"Input type: {input_details[0]['dtype']}\")\n"
            "print(f\"Output shape: {output_details[0]['shape']}\")\n"
            "print(f\"Output type: {output_details[0]['dtype']}\")"
        ]
    })
    
    # Arduino code generation section
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": "## 9. Generate Arduino Code for ESP32-S3"
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Convert TFLite model to C header file\n"
            "def tflite_to_header(tflite_model, header_name=\"model_data\"):\n"
            "    \"\"\"\n"
            "    Convert TFLite model to C header file for Arduino\n"
            "    \"\"\"\n"
            "    model_size = len(tflite_model)\n"
            "    \n"
            "    header_content = f'''\n"
            "#ifndef {header_name.upper()}_H_\n"
            "#define {header_name.upper()}_H_\n\n"
            "// Elderly Fall Detection Model\n"
            "// Generated from TensorFlow Lite model\n"
            "// Model size: {model_size} bytes\n\n"
            "const unsigned int {header_name}_len = {model_size};\n"
            "const unsigned char {header_name}[] = {{\n"
            "'''\n"
            "    \n"
            "    # Convert model bytes to hex format\n"
            "    hex_data = []\n"
            "    for i, byte in enumerate(tflite_model):\n"
            "        if i % 16 == 0:\n"
            "            hex_data.append('\\n  ')\n"
            "        hex_data.append(f'0x{byte:02x}, ')\n"
            "    \n"
            "    header_content += ''.join(hex_data)[:-2]  # Remove last comma and space\n"
            "    header_content += f'''\n"
            "}};\n\n"
            "#endif  // {header_name.upper()}_H_\n"
            "'''\n"
            "    \n"
            "    return header_content\n\n"
            "# Convert model to header file\n"
            "header_content = tflite_to_header(tflite_model, \"fall_detection_model\")\n\n"
            "with open('fall_detection_model.h', 'w') as f:\n"
            "    f.write(header_content)\n\n"
            "print(f\"Model header file saved as 'fall_detection_model.h'\")\n"
            "print(f\"Model size: {len(tflite_model)} bytes ({len(tflite_model)/1024:.2f} KB)\")"
        ]
    })
    
    # Complete Arduino code generation
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Generate complete Arduino code for ESP32-S3\n"
            "arduino_code_template = '''\n"
            "/*\n"
            "  Elderly Fall Detection using CNN-LSTM with Attention\n"
            "  ESP32-S3 Implementation\n"
            "  \n"
            "  This code implements a fall detection system using the trained model.\n"
            "  The model processes accelerometer and gyroscope data to detect falls.\n"
            "*/\n\n"
            "#include <TensorFlowLite_ESP32.h>\n"
            "#include <tensorflow/lite/micro/all_ops_resolver.h>\n"
            "#include <tensorflow/lite/micro/micro_error_reporter.h>\n"
            "#include <tensorflow/lite/micro/micro_interpreter.h>\n"
            "#include <tensorflow/lite/schema/schema_generated.h>\n"
            "#include <tensorflow/lite/version.h>\n"
            "#include <Wire.h>\n"
            "#include <MPU6050.h>\n"
            "#include \"fall_detection_model.h\"\n\n"
            "// Model and inference constants\n"
            "const int SEQUENCE_LENGTH = {sequence_length};\n"
            "const int FEATURES_PER_STEP = {features_per_step};\n"
            "const int NUM_CLASSES = {num_classes};\n"
            "const int TENSOR_ARENA_SIZE = 70000;\n\n"
            "// Activity labels\n"
            "const char* ACTIVITY_LABELS[] = {{\n"
            "{class_labels}\n"
            "}};\n\n"
            "// Sensor and model variables\n"
            "MPU6050 mpu;\n"
            "float sensor_buffer[SEQUENCE_LENGTH][FEATURES_PER_STEP];\n"
            "int buffer_index = 0;\n"
            "bool buffer_full = false;\n\n"
            "// TensorFlow Lite variables\n"
            "tflite::MicroErrorReporter micro_error_reporter;\n"
            "tflite::AllOpsResolver resolver;\n"
            "const tflite::Model* model;\n"
            "tflite::MicroInterpreter* interpreter;\n"
            "TfLiteTensor* input;\n"
            "TfLiteTensor* output;\n"
            "uint8_t tensor_arena[TENSOR_ARENA_SIZE];\n\n"
            "// Normalization parameters from training\n"
            "float feature_means[FEATURES_PER_STEP] = {{\n"
            "{feature_means}\n"
            "}};\n\n"
            "float feature_stds[FEATURES_PER_STEP] = {{\n"
            "{feature_stds}\n"
            "}};\n\n"
            "void setup() {{\n"
            "  Serial.begin(115200);\n"
            "  \n"
            "  // Initialize I2C and MPU6050\n"
            "  Wire.begin();\n"
            "  mpu.initialize();\n"
            "  \n"
            "  if (!mpu.testConnection()) {{\n"
            "    Serial.println(\"MPU6050 connection failed!\");\n"
            "    while(1);\n"
            "  }}\n"
            "  \n"
            "  Serial.println(\"MPU6050 initialized successfully\");\n"
            "  \n"
            "  // Load TensorFlow Lite model\n"
            "  model = tflite::GetModel(fall_detection_model);\n"
            "  if (model->version() != TFLITE_SCHEMA_VERSION) {{\n"
            "    Serial.println(\"Model schema version mismatch!\");\n"
            "    while(1);\n"
            "  }}\n"
            "  \n"
            "  // Create interpreter\n"
            "  interpreter = new tflite::MicroInterpreter(\n"
            "      model, resolver, tensor_arena, TENSOR_ARENA_SIZE, &micro_error_reporter);\n"
            "  \n"
            "  // Allocate memory for tensors\n"
            "  TfLiteStatus allocate_status = interpreter->AllocateTensors();\n"
            "  if (allocate_status != kTfLiteOk) {{\n"
            "    Serial.println(\"AllocateTensors() failed!\");\n"
            "    while(1);\n"
            "  }}\n"
            "  \n"
            "  // Get input and output tensors\n"
            "  input = interpreter->input(0);\n"
            "  output = interpreter->output(0);\n"
            "  \n"
            "  Serial.println(\"Fall Detection System Ready!\");\n"
            "  Serial.printf(\"Input shape: [%d, %d]\\\\n\", SEQUENCE_LENGTH, FEATURES_PER_STEP);\n"
            "  Serial.printf(\"Output classes: %d\\\\n\", NUM_CLASSES);\n"
            "}}\n\n"
            "void loop() {{\n"
            "  // Read sensor data\n"
            "  int16_t ax, ay, az, gx, gy, gz;\n"
            "  mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);\n"
            "  \n"
            "  // Convert to float and scale\n"
            "  float acc_x = ax / 16384.0; // For ±2g range\n"
            "  float acc_y = ay / 16384.0;\n"
            "  float acc_z = az / 16384.0;\n"
            "  float gyro_x = gx / 131.0;  // For ±250°/s range\n"
            "  float gyro_y = gy / 131.0;\n"
            "  float gyro_z = gz / 131.0;\n"
            "  \n"
            "  // Calculate magnitude\n"
            "  float magnitude = sqrt(acc_x*acc_x + acc_y*acc_y + acc_z*acc_z);\n"
            "  \n"
            "  // Use dummy values for temperature and altitude if not available\n"
            "  float temperature = 25.0;\n"
            "  float altitude = 100.0;\n"
            "  \n"
            "  // Store in buffer\n"
            "  sensor_buffer[buffer_index][0] = acc_x;\n"
            "  sensor_buffer[buffer_index][1] = acc_y;\n"
            "  sensor_buffer[buffer_index][2] = acc_z;\n"
            "  sensor_buffer[buffer_index][3] = magnitude;\n"
            "  sensor_buffer[buffer_index][4] = gyro_x;\n"
            "  sensor_buffer[buffer_index][5] = gyro_y;\n"
            "  sensor_buffer[buffer_index][6] = gyro_z;\n"
            "  if (FEATURES_PER_STEP > 7) {{\n"
            "    sensor_buffer[buffer_index][7] = temperature;\n"
            "  }}\n"
            "  if (FEATURES_PER_STEP > 8) {{\n"
            "    sensor_buffer[buffer_index][8] = altitude;\n"
            "  }}\n"
            "  \n"
            "  buffer_index++;\n"
            "  \n"
            "  if (buffer_index >= SEQUENCE_LENGTH) {{\n"
            "    buffer_full = true;\n"
            "    buffer_index = 0;\n"
            "  }}\n"
            "  \n"
            "  // Run inference when buffer is full\n"
            "  if (buffer_full) {{\n"
            "    runInference();\n"
            "  }}\n"
            "  \n"
            "  delay(10); // Adjust sampling rate as needed\n"
            "}}\n\n"
            "void runInference() {{\n"
            "  // Normalize and copy data to input tensor\n"
            "  for (int i = 0; i < SEQUENCE_LENGTH; i++) {{\n"
            "    for (int j = 0; j < FEATURES_PER_STEP; j++) {{\n"
            "      int index = i * FEATURES_PER_STEP + j;\n"
            "      // Apply normalization (z-score)\n"
            "      float normalized_value = (sensor_buffer[i][j] - feature_means[j]) / feature_stds[j];\n"
            "      input->data.f[index] = normalized_value;\n"
            "    }}\n"
            "  }}\n"
            "  \n"
            "  // Run inference\n"
            "  TfLiteStatus invoke_status = interpreter->Invoke();\n"
            "  if (invoke_status != kTfLiteOk) {{\n"
            "    Serial.println(\"Invoke failed!\");\n"
            "    return;\n"
            "  }}\n"
            "  \n"
            "  // Get prediction\n"
            "  float max_probability = 0;\n"
            "  int predicted_class = 0;\n"
            "  \n"
            "  for (int i = 0; i < NUM_CLASSES; i++) {{\n"
            "    float probability = output->data.f[i];\n"
            "    if (probability > max_probability) {{\n"
            "      max_probability = probability;\n"
            "      predicted_class = i;\n"
            "    }}\n"
            "  }}\n"
            "  \n"
            "  // Print results\n"
            "  Serial.printf(\"Predicted: %s (%.2f%%)\\\\n\", \n"
            "                ACTIVITY_LABELS[predicted_class], \n"
            "                max_probability * 100);\n"
            "  \n"
            "  // Special handling for fall detection\n"
            "  if (strcmp(ACTIVITY_LABELS[predicted_class], \"Falls\") == 0 && max_probability > 0.7) {{\n"
            "    Serial.println(\"*** FALL DETECTED! ***\");\n"
            "    triggerFallAlert();\n"
            "  }}\n"
            "}}\n\n"
            "void triggerFallAlert() {{\n"
            "  // Implement your fall alert mechanism here\n"
            "  Serial.println(\"Fall alert triggered!\");\n"
            "  \n"
            "  // Example: Blink built-in LED\n"
            "  pinMode(LED_BUILTIN, OUTPUT);\n"
            "  for (int i = 0; i < 5; i++) {{\n"
            "    digitalWrite(LED_BUILTIN, HIGH);\n"
            "    delay(200);\n"
            "    digitalWrite(LED_BUILTIN, LOW);\n"
            "    delay(200);\n"
            "  }}\n"
            "}}\n"
            "'''\n\n"
            "# Generate the Arduino code with actual values\n"
            "class_labels_str = ',\\n  '.join([f'  \"{cls}\"' for cls in label_encoder.classes_])\n"
            "means_str = ',\\n  '.join([f'{mean:.6f}' for mean in scaler.mean_])\n"
            "stds_str = ',\\n  '.join([f'{std:.6f}' for std in scaler.scale_])\n\n"
            "arduino_code = arduino_code_template.format(\n"
            "    sequence_length=X_train_scaled.shape[1],\n"
            "    features_per_step=X_train_scaled.shape[2],\n"
            "    num_classes=num_classes,\n"
            "    class_labels=class_labels_str,\n"
            "    feature_means=means_str,\n"
            "    feature_stds=stds_str\n"
            ")\n\n"
            "# Save Arduino code\n"
            "with open('elderly_fall_detection_esp32s3.ino', 'w') as f:\n"
            "    f.write(arduino_code)\n\n"
            "print(\"Arduino code generated and saved as 'elderly_fall_detection_esp32s3.ino'\")"
        ]
    })
    
    # Final summary section
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 10. Deployment Instructions and Summary\n\n"
            "### Files Generated:\n"
            "1. **elderly_fall_detection_esp32s3.ino** - Main Arduino sketch\n"
            "2. **fall_detection_model.h** - TensorFlow Lite model as C header\n"
            "3. **best_fall_detection_model.h5** - Keras model (for future retraining)\n"
            "4. **elderly_fall_detection_*.tflite** - TensorFlow Lite model\n\n"
            "### ESP32-S3 Setup Steps:\n"
            "1. Install ESP32 board support in Arduino IDE\n"
            "2. Install required libraries:\n"
            "   - TensorFlowLite_ESP32\n"
            "   - MPU6050 library\n"
            "   - Wire library (built-in)\n"
            "3. Connect MPU6050 to ESP32-S3:\n"
            "   - VCC → 3.3V\n"
            "   - GND → GND\n"
            "   - SDA → GPIO21 (or your preferred I2C SDA pin)\n"
            "   - SCL → GPIO22 (or your preferred I2C SCL pin)\n"
            "4. Copy all generated files to your Arduino sketch folder\n"
            "5. Upload the sketch to ESP32-S3\n\n"
            "### Customization Options:\n"
            "1. Adjust fall detection threshold in Arduino code (currently 0.7)\n"
            "2. Add additional sensors (temperature, barometer)\n"
            "3. Implement wireless alerts (WiFi, Bluetooth)\n"
            "4. Add data logging capabilities\n"
            "5. Optimize sampling rate for battery life\n\n"
            "### Notes:\n"
            "- The model is optimized for ESP32-S3 deployment\n"
            "- Fall detection threshold can be adjusted based on requirements\n"
            "- Consider adding more training data for better accuracy\n"
            "- Test thoroughly with real-world scenarios"
        ]
    })
    
    # Final summary cell
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Create final summary\n"
            "print(\"\\n\" + \"=\"*60)\n"
            "print(\"ELDERLY FALL DETECTION MODEL TRAINING COMPLETED!\")\n"
            "print(\"=\"*60)\n"
            "print(f\"Dataset: {len(X)} total samples\")\n"
            "print(f\"Model: CNN-LSTM with Attention Layer\")\n"
            "print(f\"Target: ESP32-S3 Deployment\")\n"
            "print(f\"Model Size: {len(tflite_model)/1024:.1f} KB\")\n"
            "print(f\"\\nTest Accuracy: {test_accuracy:.3f}\")\n"
            "print(f\"Test Precision: {test_precision:.3f}\")\n"
            "print(f\"Test Recall: {test_recall:.3f}\")\n"
            "print(\"\\nGenerated Files:\")\n"
            "print(\"1. elderly_fall_detection_esp32s3.ino - Arduino sketch\")\n"
            "print(\"2. fall_detection_model.h - Model header file\")\n"
            "print(\"3. best_fall_detection_model.h5 - Keras model\")\n"
            "print(f\"4. {tflite_filename} - TensorFlow Lite model\")\n"
            "print(\"\\nReady for ESP32-S3 deployment!\")\n"
            "print(\"=\"*60)"
        ]
    })
    
    # Create the complete notebook
    notebook = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {
                "gpuType": "T4",
                "provenance": []
            },
            "kernelspec": {
                "display_name": "Python 3",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 0
    }
    
    return notebook

# Create and save the notebook
notebook = create_complete_notebook()
with open('elderly_fall_detection_cnn_lstm_attention.ipynb', 'w') as f:
    json.dump(notebook, f, indent=2)

print("Complete Jupyter notebook created successfully!")
print("Notebook: elderly_fall_detection_cnn_lstm_attention.ipynb")
print("\nFeatures:")
print("- CNN-LSTM with Attention Layer for fall detection")
print("- Optimized for ESP32-S3 deployment") 
print("- Complete training pipeline")
print("- Model conversion to TensorFlow Lite")
print("- Arduino code generation")
print("- Preprocessing parameter extraction")
print("- Ready for Google Colab execution")

