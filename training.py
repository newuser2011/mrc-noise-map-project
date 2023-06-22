import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
from model import Model

np.random.seed(42)
df = pd.read_csv("with_updates/data2.csv")
d = pd.read_csv("mldata_update.csv")


# extract data -
X = []
y = []
X_train = []
X_val = []
y_train = []
y_val = []
X_test = []
y_test = []

for i in range(len(df)):
    a = np.array(d.iloc[i])

    X.append(a)
    y.append(df["TL"][i])
    if i % 10000 == 0:
        print(i, "Done")


X = np.array(X)
print(X.shape)


y = np.array(y)
# Normalize the data -
for i in range(X.shape[1]):
    mean = np.mean(X[:, i])
    std = np.std(X[:, i])
    X[:, i] = (X[:, i] - mean)/(std + 1e-4)


print("Data extracted")
print("----------------")

indices = np.arange(len(X))
shuffled_indices = np.random.permutation(indices)

for i in range(int(0.6 * len(shuffled_indices))):
    X_train.append(X[shuffled_indices[i]])
    y_train.append(y[shuffled_indices[i]])

X_train = np.array(X_train)
y_train = np.array(y_train)

print("Train Data Created")
print("--------------------")

for i in range(int(0.6 * len(shuffled_indices)), int(0.8 * len(shuffled_indices))):
    X_val.append(X[shuffled_indices[i]])
    y_val.append(y[shuffled_indices[i]])

X_val = np.array(X_val)
y_val = np.array(y_val)

print("Val Data Created")
print("--------------------")

for i in range(int(0.8 * len(shuffled_indices)), int(len(shuffled_indices))):
    X_test.append(X[shuffled_indices[i]])
    y_test.append(y[shuffled_indices[i]])

print("Test Data Created")
print("--------------------")

X_test = np.array(X_test)
y_test = np.array(y_test)

y_train = y_train.reshape(-1, 1)
y_val = y_val.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

epochs = 200
batch_size = 128  # 256
learning_rate = 5e-4
x = len(X_train)//batch_size
train_losses = []
val_losses = []

model = Model(X_train[0].shape[-1], 256, 256, 64, 32)  # 64, 32, 32, 16

print("Starting Training")
print("------------------")
for i in range(epochs):
    loss = 0
    try:
        for j in range(x):
            model.forward(X_train[j * batch_size: (j+1) * batch_size], p=0.8)
            model.compute_loss(y_train[j * batch_size: (j+1) * batch_size])
            loss = model.loss
            model.backward(X_train[j * batch_size: (j+1) * batch_size], y_train[j * batch_size : (j+1) * batch_size])
            model.sgd_update(lr=learning_rate)  # 1e-3

        train_losses.append(loss)

        model.forward(X_val, 1)
        model.compute_loss(y_val)
        losses = model.loss
        val_losses.append(losses)
    except KeyboardInterrupt:
        print(f"Training stopped at Epoch: {i} with train loss: {train_losses[-1]}, val loss: {val_losses[-1]}")
        break

    print("Epoch -> ", i, "Mean Train Loss -> ", loss, "val loss -> ", losses)


print("Finished Training")
print("------------------")

print("z1", model.z1)
print("z2", model.z2)
print("z3", model.z3)
print("z4", model.z4)
print("z5", model.z5)
#print("z6", model.z6)
#print("z7", model.z7)
# print("z8", model.z8)
# print("z9", model.z9)

# Test on the test data -
model.forward(X_test, p = 1)
model.compute_loss(y_test)
test_loss = model.loss
print("Test Loss", test_loss)

plt.plot(train_losses, label = "Training loss")
plt.plot(val_losses, label = "Validation loss")
plt.legend(loc = "best")
plt.show()


# Assuming the model trained till now, #lets save the predictions
model.forward(X, p = 1)
preds = model.z5

errors = np.array(df["TL"]).reshape(-1, 1) - preds
plt.plot(df["Distance"], errors)
print(f"Percentage of pairs having error more than 3dB: {np.mean(np.abs(errors) > 3)}")
print(f"Percentage of pairs having error more than 5dB: {np.mean(np.abs(errors) > 5)}")
print(f"Percentage of pairs having error more than 9dB: {np.mean(np.abs(errors) > 9)}")
df["NN TL"] = preds
df.to_csv("with_updates/data2.csv", index=False)