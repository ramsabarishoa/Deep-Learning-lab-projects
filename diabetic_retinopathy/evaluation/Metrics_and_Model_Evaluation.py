acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

"""img = "/content/drive/MyDrive/IDRID_dataset/images/test/IDRiD_070.jpg"


img = tf.io.read_file(img)
img = tf.io.decode_jpeg(img)
img = tf.cast(img,tf.float32) / 255
img = tf.image.resize_with_pad(img,img_ht,img_wd,antialias=True)
img = tf.reshape(img, [1,256,256,3])
plt.imshow(np.squeeze(img.numpy()))
x = model.predict(img)
print(x)
label = np.argmax(x)
print("Label" ,label)
plt.show()"""

#Prediction
#Loading test images 

test_images = glob.glob("/content/drive/MyDrive/IDRID_dataset/images/test/*.jpg")
#print(len(test_images))
df_test = pd.read_csv('/content/drive/MyDrive/IDRID_dataset/labels/test.csv')
df_test['Retinopathy grade'] = df_test['Retinopathy grade'].replace([0,1,2,3,4],[0,0,1,1,1])
df_test['Image name'] = df_test['Image name'] + '.jpg'
df_test = df_test.drop_duplicates()
df_test = df_test.iloc[:, : 2]
#print(df_test)
predicted_label_list = []
for iname,iclass in df_test.itertuples(index=False):
    for file in train_images:
      if os.path.basename(file) == iname:
        img = tf.io.read_file(file)
        img = tf.io.decode_jpeg(img)
        img = tf.cast(img,tf.float32) / 255
        img = tf.image.resize_with_pad(img,img_ht,img_wd,antialias=True)
        img = tf.reshape(img, [1,256,256,3])
        x = model.predict(img)
        predicted_label = np.argmax(x)
        predicted_label_list.append(predicted_label)

df_test['Predicted Class'] = predicted_label_list
df_test['Result'] = np.where(df_test['Retinopathy grade'] == df_test['Predicted Class'], 'Correct Prediction', 'Incorrect Prediction')
print(df_test)

df_test['Result'].value_counts()

from sklearn import metrics
from sklearn.metrics import confusion_matrix
import seaborn as sb
cm = confusion_matrix(df_test['Retinopathy grade'],df_test['Predicted Class'])
plt.figure(figsize = (10,5))
plt.title('Confusion Matrix')
sb.heatmap(cm, cmap="Blues", annot=True,annot_kws={"size": 16})
print('Test Accuracy:',metrics.accuracy_score(df_test['Retinopathy grade'], df_test['Predicted Class']))

