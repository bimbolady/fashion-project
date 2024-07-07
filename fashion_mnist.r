# R code for classifying images using profileimages to target marketing for different products

install.packages("keras")
library(keras)

# Load the Fashion MNIST dataset
fashion_mnist <- dataset_fashion_mnist()
c(x_train, y_train) %<-% fashion_mnist$train
c(x_test, y_test) %<-% fashion_mnist$test

# Preprocess the Data
# Normalize the images to the range [0, 1]
x_train <- x_train / 255
x_test <- x_test / 255

# Reshape the data to include the channel dimension
x_train <- array_reshape(x_train, c(nrow(x_train), 28, 28, 1))
x_test <- array_reshape(x_test, c(nrow(x_test), 28, 28, 1))

# One-hot encode the labels
y_train <- to_categorical(y_train, 10)
y_test <- to_categorical(y_test, 10)

model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3,3), activation = 'relu', input_shape = c(28, 28, 1)) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3,3), activation = 'relu') %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filters = 128, kernel_size = c(3,3), activation = 'relu') %>%
  layer_flatten() %>%
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 10, activation = 'softmax')

# Compile the model
model %>% compile(
  optimizer = 'adam',
  loss = 'categorical_crossentropy',
  metrics = 'accuracy'
)

# Print the model summary
summary(model)

# Training the model
history <- model %>% fit(
  x_train, y_train,
  epochs = 10, batch_size = 64,
  validation_split = 0.2
)

# Evaluating the model
score <- model %>% evaluate(x_test, y_test)
cat('Test accuracy:', score$acc, '\n')

# Predictions
# Predicting two images from the test set
predictions <- model %>% predict(x_test[1:2,,,])

# Display the predictions and the corresponding images
par(mfrow=c(1, 2))
for (i in 1:2) {
  image(x_test[i,,,], col = gray.colors(256), main = paste('Predicted:', which.max(predictions[i,]) - 1, 'True:', which.max(y_test[i,]) - 1))
}
