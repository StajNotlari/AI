import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

import inception

inception.download()

def conv_layer_names ():
    model = inception.Inception()
    names = [op.name for op in model.graph.get_operations() if op.type=='Conv2D']
    model.close()
    return names

conv_names = conv_layer_names()

def optimize_image (conv_id=0, feature=0, iterations=30, show_progress=True):
    model = inception.Inception()
    resized_image = model.resized_image

    conv_name = conv_names[conv_id]
    tensor = model.graph.get_tensor_by_name(conv_name + ":0")

    with model.graph.as_default():
        loss = tf.reduce_mean(tensor[:, :, :, feature])

    gradient = tf.gradients(loss, resized_image)
    sess = tf.Session(graph=model.graph)

    image_shape = resized_image.get_shape()
    image = np.random.uniform(size=image_shape) + 128.0

    for i in range (iterations):
        feed_dict  = {model.tensor_name_resized_image: image}
        [grad, loss_value] = sess.run([gradient, loss], feed_dict=feed_dict)
        grad = np.array(grad).squeeze()

        step_size = 1.0 / (grad.std() + 1e-8)
        image += step_size * grad
        image = np.clip(image, 0.0, 255.0)

        if show_progress:
            print("Iteration:", i)
            msg = "Gradient min: {0:>9.6f}, max: {1:>9.6f}, stepsize: {2:>9.2f}"
            print(msg.format(grad.min(), grad.max(), step_size))
            print("Loss:", loss_value)
            print()

    model.close()
    return image.squeeze()

def normalize_image (x):
    x_min = x.min()
    x_max = x.max()

    x_norm = (x - x_min) / (x_max - x_min)

    return x_norm

def plot_image (image):
    img_norm = normalize_image(image)
    plt.imshow(img_norm)
    plt.show()

image = optimize_image(conv_id=75, feature=10, iterations=2000)
plot_image(image)


