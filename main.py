import logging
from logging.handlers import MemoryHandler, RotatingFileHandler

import numpy as np
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.DEBUG)
log_file = 'logs/log.txt'
max_log_size = 1024 * 1024 * 50  # 50 MB
file_handler = RotatingFileHandler(log_file, maxBytes=max_log_size, backupCount=1)
memory_handler = MemoryHandler(1024 * 5, flushLevel=logging.ERROR, target=file_handler)
logging.getLogger().addHandler(memory_handler)

np.random.seed(42)


def main():
    # Read training images
    train_images = read_images('dataset/train-images.idx3-ubyte')

    # Read training labels
    train_labels = read_labels('dataset/train-labels.idx1-ubyte')

    # Read test images
    test_images = read_images('dataset/t10k-images.idx3-ubyte')

    # Read test labels
    test_labels = read_labels('dataset/t10k-labels.idx1-ubyte')

    # Flatten the images (magic code dont touch)
    train_images = train_images.reshape(train_images.shape[0], -1)
    test_images = test_images.reshape(test_images.shape[0], -1)

    # some info about the layers
    input_size = 256
    hidden_size = 128
    output_size = 10

    # Initialized Weights and biases for the layers
    weights_input_hidden = np.random.randn(input_size, hidden_size) * np.sqrt(2 / (input_size + hidden_size))
    biases_input_hidden = np.random.randn(hidden_size) * 0.01
    logging.debug(f"Initial weights_input_hidden: {weights_input_hidden}")
    logging.debug(f"Initial biases_input_hidden: {biases_input_hidden}")

    weights_hidden_output = np.random.randn(hidden_size, output_size) * np.sqrt(2 / (hidden_size + output_size))
    biases_hidden_output = np.random.randn(output_size) * 0.01  # end weights and biases
    logging.debug(f"Initial weights_hidden_output: {weights_hidden_output}")
    logging.debug(f"Initial biases_hidden_output: {biases_hidden_output}")

    num_epochs = 10
    learning_rate = 0.0003
    for epoch in range(num_epochs):
        total_loss = 0
        logging.debug(f"Epoch: {epoch + 1}, Learning rate: {learning_rate}")

        for image, label in zip(train_images, train_labels):
            # Make -> forward propagate -> output
            input_layer = np.array(image)
            hidden_layer = forward_propagate(input_layer, weights_input_hidden, biases_input_hidden)
            output_layer = forward_propagate(hidden_layer, weights_hidden_output, biases_hidden_output)

            # it's the index of the element with the highest confidence (value)
            prediction = np.argmax(output_layer) + 1  # +1 cause that is the index not the actual number

            # Calculate how wrong its ass was in predicting the label
            loss = compute_loss(prediction, label)
            total_loss += loss

            # Backward propagate the error
            expected_output = np.zeros_like(output_layer)
            expected_output[label - 1] = 1  # Set the target class index to 1
            error_output = output_layer - expected_output  # Calculate the error in the output layer

            # For debug purposes
            # old_weights_hidden_output = weights_hidden_output
            # old_biases_hidden_output = biases_hidden_output
            # old_weights_input_hidden = weights_input_hidden
            # old_biases_input_hidden = biases_input_hidden

            # Calculate the gradients of the weights and biases of the hidden - output layer
            grad_weights_hidden_output, grad_biases_hidden_output = calculate_gradients(hidden_layer,
                                                                                        biases_hidden_output,
                                                                                        error_output)

            # calculate the gradients for the input-to-hidden layer, propagating the error (d_output) from the output layer
            # back to the hidden layer
            error_hidden = np.dot(error_output, weights_hidden_output.T)
            grad_weights_input_hidden, grad_biases_input_hidden = calculate_gradients(input_layer, biases_input_hidden,
                                                                                      error_hidden)

            weights_hidden_output -= learning_rate * grad_weights_hidden_output  # Update weights between hidden and output layers
            biases_hidden_output -= learning_rate * grad_biases_hidden_output  # Update biases in the output layer
            weights_input_hidden -= learning_rate * grad_weights_input_hidden  # Update weights between input and hidden layers
            biases_input_hidden -= learning_rate * grad_biases_input_hidden  # Update biases in the hidden layer

            if epoch == -500:  # Specify the epochs you want to log
                logging.debug(f"Epoch: {epoch + 1}")

                logging.debug(f"Input layer: {input_layer}")
                logging.debug(f"Hidden layer: {hidden_layer}")
                logging.debug(f"Output layer: {output_layer}")

                logging.debug(f"Prediction: {prediction}")
                logging.debug(f"Loss: {loss}")

                # logging.debug(f"Old weights_input_hidden: {old_weights_input_hidden}")
                # logging.debug(f"Old biases_input_hidden: {old_biases_input_hidden}")
                # logging.debug(f"Old weights_hidden_output: {old_weights_hidden_output}")
                # logging.debug(f"Old biases_hidden_output: {old_biases_hidden_output}")

                logging.debug(f"Biases Gradient hidden-output: {grad_biases_hidden_output}")
                logging.debug(f"weights Gradient hidden-output: {grad_weights_hidden_output}")
                logging.debug(f"Biases Gradient input-hidden: {grad_biases_hidden_output}")
                logging.debug(f"Weights Gradient input-hidden: {grad_weights_hidden_output}")

                logging.debug(f"Updated weights_input_hidden: {weights_input_hidden}")
                logging.debug(f"Updated biases_input_hidden: {biases_input_hidden}")
                logging.debug(f"Updated weights_hidden_output: {weights_hidden_output}")
                logging.debug(f"Updated biases_hidden_output: {biases_hidden_output}")

                logging.debug("-" * 100)

            # print(f"{epoch}:{i} {output_layer}\n {expected_output}\n {prediction}\n{label}\n {loss}\n\n")

        average_loss = total_loss / len(train_labels)

        # Test on testing set
        test_predictions = []
        for image in test_images:
            input_layer = image
            hidden_layer = forward_propagate(input_layer, weights_input_hidden, biases_input_hidden)
            output_layer = forward_propagate(hidden_layer, weights_hidden_output, biases_hidden_output)

            # it's the index of the element with the highest confidence (value)
            test_predictions.append(np.argmax(output_layer) + 1)  # +1 cause that is the index not the actual number

        accuracy = calculate_accuracy(test_predictions, test_labels)
        print(f"{epoch + 1}/{num_epochs}: avg Loss: {average_loss}, accuracy: {accuracy}%")
    print(f"Input-Hidden Weights {weights_input_hidden}\n Hidden-Output Weights {weights_hidden_output}\n"
          f" Input-Hidden Biases {biases_input_hidden}\n Hidden-Output Biases{biases_hidden_output}")


def calculate_accuracy(predictions, labels):
    correct = np.sum(predictions == labels)  # Count the number of correct predictions
    total = len(predictions)  # Total number of predictions
    accuracy = correct / total * 100.0  # Calculate the accuracy as a percentage
    return accuracy


def softmax(x):  # Magic code again
    e_x = np.exp(x - np.max(x))  # Subtract max value from it to avoid overflow
    norm_x = e_x / np.sum(e_x)
    return norm_x


def calculate_gradients(in_layer, in_biases, error):
    # Backward propagate the error (all of this is magic i have no idea whats going on)

    grad_weights = np.outer(in_layer, error)  # Gradient of weights between hidden and output layers
    grad_biases = np.dot(in_biases, error)  # Gradient of biases in the output layer

    return grad_weights, grad_biases


def compute_loss(prediction, reality):
    loss = np.mean((prediction - reality) ** 2)  # magic calculation (MSE)
    return loss


def forward_propagate(in_layer, weights, biases):
    # basically multiply each element in in_layer with its weight(i.e. connection between the two neurons) and then
    # add on the bias of the neuron
    weighted_input = np.dot(in_layer, weights) + biases
    return activation(weighted_input)  # pass it thru the activation function (RelU) to get the output of this layer


def relu_derivative(inputs):  # Derivative of ReLU activation function i.e. 1 if input > 0 else 0
    result = np.zeros_like(inputs)
    for i, inp in enumerate(inputs):
        if inp >= 0:
            result[i] = 1
        else:
            result[i] = 0

    return inputs


def activation(w_inputs):
    activated = np.empty_like(w_inputs)  # create another array with same shape as w_input

    for i, w_input in enumerate(w_inputs):  # Perform RelU on each element
        if w_input <= 0:
            activated[i] = 0
        else:
            activated[i] = w_input

    return softmax(activated)  # normalize the value


def resize_images(images, new_size):  # magic again
    resized_images = []
    for img in images:
        img_pil = Image.fromarray(img)
        resized_img = img_pil.resize(new_size, resample=Image.BILINEAR)
        resized_images.append(np.array(resized_img))

    return np.array(resized_images)


def read_images(filename):
    with open(filename, 'rb') as f:
        # Read the file headers
        _ = int.from_bytes(f.read(4), 'big')
        num_images = int.from_bytes(f.read(4), 'big')
        num_rows = int.from_bytes(f.read(4), 'big')
        num_cols = int.from_bytes(f.read(4), 'big')

        # Read the image data
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(num_images, num_rows, num_cols)

    return resize_images(images, (16, 16))


def read_labels(filename):
    with open(filename, 'rb') as f:
        # Read the file headers
        _ = int.from_bytes(f.read(4), 'big')
        num_labels = int.from_bytes(f.read(4), 'big')

        # Read the label data
        labels = np.frombuffer(f.read(), dtype=np.uint8)

    return labels


if __name__ == '__main__':
    main()
