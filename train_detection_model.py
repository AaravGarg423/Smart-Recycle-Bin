import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os

# === SETTINGS ===
DATASET_DIR = r"C:\Aarav Things\MLProject\dataset"
IMG_SIZE = (128, 128)
BATCH_SIZE = 16
EPOCHS = 30
MODEL_SAVE_PATH = r"C:\Aarav Things\MLProject\BottleClassifier.keras"

# Check data
bottle_path = os.path.join(DATASET_DIR, "Bottle")
notbottle_path = os.path.join(DATASET_DIR, "NotBottle")

bottle_count = len([f for f in os.listdir(bottle_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
notbottle_count = len([f for f in os.listdir(notbottle_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

print(f"[INFO] Bottle images: {bottle_count}")
print(f"[INFO] NotBottle images: {notbottle_count}")

if bottle_count < 50 or notbottle_count < 50:
    print("[ERROR] Need at least 50 images in each folder!")
    exit()

# === DATA GENERATORS ===
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.15,
    brightness_range=[0.8, 1.2]
)

val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

# FORCE ORDER: Bottle=0, NotBottle=1
train_gen = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training',
    shuffle=True,
    classes=['Bottle', 'NotBottle']
)

val_gen = val_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation',
    classes=['Bottle', 'NotBottle']
)

print(f"[INFO] Class indices: {train_gen.class_indices}")
print(f"[INFO] Training samples: {train_gen.samples}")
print(f"[INFO] Validation samples: {val_gen.samples}")

# === TRANSFER LEARNING MODEL ===
print("[INFO] Loading pre-trained MobileNetV2...")

# Load MobileNetV2 pre-trained on ImageNet
base_model = MobileNetV2(
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    include_top=False,
    weights='imagenet'
)

# Freeze base model
base_model.trainable = False

# Add custom layers
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("[INFO] Model architecture:")
model.summary()

# === CALLBACKS ===
checkpoint = ModelCheckpoint(
    MODEL_SAVE_PATH,
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

# === TRAIN ===
print("\n[INFO] Starting training (Phase 1: Frozen base)...")
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=15,
    callbacks=[checkpoint, early_stop]
)

# === FINE-TUNE (Unfreeze some layers) ===
print("\n[INFO] Fine-tuning (Phase 2: Unfrozen top layers)...")
base_model.trainable = True

# Freeze all except last 20 layers
for layer in base_model.layers[:-20]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),  # Lower learning rate
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history_fine = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=15,
    callbacks=[checkpoint, early_stop]
)

# === SAVE FINAL MODEL ===
model.save(MODEL_SAVE_PATH)
print(f"\n[INFO] ✓ Model saved to {MODEL_SAVE_PATH}")
print(f"[INFO] Final validation accuracy: {history_fine.history['val_accuracy'][-1]:.3f}")

# === TEST THE MODEL ===
print("\n[INFO] Testing model predictions...")
import numpy as np

# Get a sample from validation
sample_images, sample_labels = next(val_gen)

predictions = model.predict(sample_images[:5])
print("\nSample predictions (should vary between 0 and 1):")
for i, (pred, label) in enumerate(zip(predictions, sample_labels[:5])):
    print(f"  Image {i+1}: pred={pred[0]:.4f}, actual={'Bottle' if label==0 else 'NotBottle'}")


## **RUN THIS:**

