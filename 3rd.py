
#import the required libraries

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential 
from keras.layers import Conv2D,Dense,Dropout,Flatten,MaxPooling2D




mnist=tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()
input_shape=(28,28,1)
#print(x_train.shape)

x_train=x_train.reshape(x_train.shape[0],28,28,1);
x_test=x_test.reshape(x_test.shape[0],28,28,1)
#print(x_train.shape)

x_train=(x_train-0.0)/(255.0-0.0)
x_test=(x_test-0.0)/(255-0.0)

#print(x_train[0].max())
#print(x_test[0].min())

#print(x_train.shape)
#print(x_test.shape)



model=Sequential()
model.add(Conv2D(28,kernel_size=(3,3),input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=2))
model.add(Flatten())
model.add(Dense(200,activation="relu"))
model.add(Dropout(.3))
model.add(Dense(10,activation="softmax"));

model.summary()





# 3- training the model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=['accuracy']
)

model.fit(x_train,y_train,epochs=2)






# 4 -estimating the model
test_loss,test_acc=model.evaluate(x_test,y_test)
print("test_loss ",test_loss);
print("Test_acc ",test_acc)


image=x_train[1]
plt.imshow(np.squeeze(image),cmap='gray')
plt.show()

image=image.reshape(1,image.shape[0],image.shape[1],image.shape[2])
predict_model=model.predict([image])
np.argmax(predict_model)


