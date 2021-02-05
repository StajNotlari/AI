######Bu hangi katman neye odaklanıyor onu görmek için
######ilk katmanlar ve eğitimin başlarında basit resimler geliyor. yani cizgiye yada çapraz desene vs odaklanıyor
######ilerki katmanve eğitiminz sonlarında çok daha kompleks sonuçlar çıkıyor.

import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

import numpy as np

import inception

inception.download()

def conv_layer_names ():
    model = inception.Inception()
    names = [op.name for op in model.graph.get_operations() if op.type=='Conv2D']
    model.close()
    return names

conv_names = conv_layer_names() #94 katman
#print(len(conv_names))

def show_image(image):
    plt.imshow(image)
    plt.show()

def optimize_image (conv_id=0, feature=0, iterations=30, show_progress=True):
    model = inception.Inception()
    resized_image = model.resized_image #Tensor("ResizeBilinear:0", shape=(1, 299, 299, 3), dtype=float32)
    #print(resized_image)

    conv_name = conv_names[conv_id]
    layer = model.graph.get_tensor_by_name(conv_name + ":0") #conv_id=75 #Tensor("mixed_8/tower_1/conv_3/Conv2D:0", shape=(1, 8, 8, 192), dtype=float32)

    with model.graph.as_default(): #operator ekleyebilmek için modelin grafiğini default yapıyoruz dedi
        loss = tf.reduce_mean(layer[:, :, :, feature]) #loss fonksiyonu belirtilen feature için tüm tensor değerlerinin ortalaması dedi

    gradient = tf.gradients(loss, resized_image)
    sess = tf.Session(graph=model.graph)

    image_shape = resized_image.get_shape()  # boş resim ver

    image_shape = resized_image.get_shape() #boş resim ver
    image = np.random.uniform(size=image_shape) + 128.0 #random doldur


    #örnek resmi göster diye
    #temp_image = np.random.uniform(size=image_shape)
    #show_image(temp_image.reshape([299, 299, 3]) * 255.)



    for i in range (iterations):
        feed_dict  = {model.tensor_name_resized_image: image}
        [grad, loss_value] = sess.run([gradient, loss], feed_dict=feed_dict)

        #print(grad[0].shape, loss_value) #(1, 299, 299, 3) -0.42605704

        grad = np.array(grad).squeeze()#np array a sıkıştırıyoruz dedi. aşağı satıra bakılırsa dizinin başındaki 1 gitmiş

        #print(grad.shape)#(299, 299, 3)

        #print(grad.std())#6.330782e-05

        step_size = 1.0 / (grad.std() + 1e-8)
        image += step_size * grad
        image = np.clip(image, 0.0, 255.0)#resmin geçerli renk aralığında olduğundan emin oluyoruz

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

#image = optimize_image(conv_id=1, feature=1, iterations=2)
image = optimize_image(conv_id=80, feature=10, iterations=501)
plot_image(image)

