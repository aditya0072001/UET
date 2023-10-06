# handwritten digits classification

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm
from sklearn import metrics

# load the digits dataset
digits = datasets.load_digits()

# display the first digit
#plt.figure(1, figsize=(3, 3))
#plt.imshow(digits.images[0], cmap=plt.cm.gray_r, interpolation='nearest')
#plt.show()

# display the first 15 digits

images_and_labels = list(zip(digits.images, digits.target))
plt.figure(1, figsize=(15, 15))
for index, (image, label) in enumerate(images_and_labels[:15]):
    plt.subplot(3, 5, index+1)
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Training: %i' % label)


plt.show()

# flatten the images

n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

# create a classifier: a support vector classifier

classifier = svm.SVC(gamma=0.001)

# we learn the digits on the first half of the digits

classifier.fit(data[:n_samples//2], digits.target[:n_samples//2])

# now predict the value of the digit on the second half

expected = digits.target[n_samples//2:]

predicted = classifier.predict(data[n_samples//2:])

print("Classification report for classifier %s:\n%s\n"
        % (classifier, metrics.classification_report(expected, predicted)))

print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

# display the first 4 test digits with their predicted labels

images_and_predictions = list(zip(digits.images[n_samples//2:], predicted))
plt.figure(1, figsize=(15, 15))
for index, (image, prediction) in enumerate(images_and_predictions[:4]):
    plt.subplot(3, 5, index+1)
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Prediction: %i' % prediction)

plt.show()

# display the last 4 test digits with their predicted labels

images_and_predictions = list(zip(digits.images[n_samples//2:], predicted))
plt.figure(1, figsize=(15, 15))
for index, (image, prediction) in enumerate(images_and_predictions[-4:]):
    plt.subplot(3, 5, index+1)
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Prediction: %i' % prediction)

plt.show()




