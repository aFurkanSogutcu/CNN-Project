from keras.datasets import cifar100
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D,Dense, Dropout,  Activation, GlobalAveragePooling2D
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
import numpy as np
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')

classes = [17, 22, 53, 55, 60, 71, 90]
class_map = {k: i for i, k in enumerate(classes)}

y_train_mapped = np.array([class_map[y[0]] for y in y_train if y[0] in classes])
y_test_mapped = np.array([class_map[y[0]] for y in y_test if y[0] in classes])

x_train_filtered = np.array([x for i, x in enumerate(x_train) if y_train[i][0] in classes])
x_test_filtered = np.array([x for i, x in enumerate(x_test) if y_test[i][0] in classes])

# Girdileri normalleştirme
x_train_last = x_train_filtered.astype('float32') / 255
x_test_last = x_test_filtered.astype('float32') / 255

y_train_categorical = to_categorical(y_train_mapped, num_classes=len(classes))
y_test_categorical = to_categorical(y_test_mapped, num_classes=len(classes))

model = Sequential([
    Conv2D(32, (3, 3), padding='same', input_shape=(32, 32, 3)),
    Activation('relu'),
    MaxPooling2D((2, 2)),
    
    Conv2D(64, (3, 3), padding='same'),
    Activation('relu'),
    MaxPooling2D((2, 2)),
    
    Conv2D(128, (3, 3), padding='same'),
    Activation('relu'),
    GlobalAveragePooling2D(),
    
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(classes), activation='softmax')
])

early_stopping = EarlyStopping(monitor='val_loss', patience=10)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Modeli eğit
history = model.fit(x_train_last, y_train_categorical, epochs=200, validation_data=(x_test_last, y_test_categorical), batch_size=50,callbacks=[early_stopping])

# Modeli değerlendir
loss, accuracy = model.evaluate(x_test_last, y_test_categorical)
print(f'Test Doğruluğu: {accuracy*100:.2f}%')

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='validation accuracy')
plt.title('train and validation accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='validation loss')
plt.title('train and test loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()

plt.show()