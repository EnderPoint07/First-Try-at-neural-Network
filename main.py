import numpy as np
from PIL import Image


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

    # Weights and biases for the layers
    weights_input_hidden = np.random.randn(input_size, hidden_size) * 0.01
    biases_input_hidden = np.zeros(hidden_size)

    weights_hidden_output = np.random.randn(hidden_size, output_size) * 0.01
    biases_hidden_output = np.zeros(output_size)  # end weights and biases

    # Make -> forward propagate -> output (done using  my proud function process_layer())
    input_layer = train_images[0]
    hidden_layer = forward_propagate(input_layer, weights_input_hidden, biases_input_hidden)
    output_layer = forward_propagate(hidden_layer, weights_hidden_output, biases_hidden_output)

    # it's the index of the element with the highest confidence (value)
    prediction = np.argmax(output_layer) + 1  # +1 cause that is the index not the actual number

    # Calculate how wrong its ass was in predicting the label
    loss = compute_loss(prediction, train_labels[0])

    print(f"{output_layer}\n {prediction}\n {train_labels[0]}\n {loss}")


def compute_loss(prediction, reality):
    loss = np.mean((prediction - reality) ** 2)  # magic calculation (MSE)
    return loss


def forward_propagate(in_layer, weights, biases):
    # basically multiply each element in in_layer with its weight(i.e. connection between the two neurons) and then
    # add on the bias of the neuron
    weighted_input = np.dot(in_layer, weights) + biases
    return activation(weighted_input)  # pass it thru the activation function (RelU) to get the output of this layer


def activation(w_inputs):
    activated = np.empty_like(w_inputs)  # create another array with same shape as w_input

    for i, w_input in enumerate(w_inputs):  # Perform RelU on each element
        if w_input <= 0:
            activated[i] = 0
        else:
            activated[i] = w_input

    return activated


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
