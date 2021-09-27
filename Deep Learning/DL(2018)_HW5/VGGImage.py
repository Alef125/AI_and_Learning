from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model


import matplotlib.pyplot as plt

model = VGG16(include_top=True, weights='imagenet',
              input_tensor=None, input_shape=None,
              pooling='max', classes=1000)

plt.figure(figsize=(30, 30))

pic_names = ['brown_bear', 'cat_dog', 'dd_tree', 'dog_beagle', 'scenery', 'space_shuttle']

layer_dict = dict([(layer.name, layer) for layer in model.layers])

for pic_name in pic_names:

    image_dir = 'Images/' + pic_name + '.png'
    image = load_img(image_dir, target_size=(224, 224))
    image = img_to_array(image)
    shape = image.shape
    new_shape = (1, shape[0], shape[1], shape[2])
    image = image.reshape(new_shape)
    image = preprocess_input(image)
    y = model.predict(image)
    label = decode_predictions(y)
    label = label[0][0]
    print pic_name, 'is predicted as:', label[1], 'with belief:', label[2] * 100, '%'

    layer_output_layer_3 = layer_dict['block1_pool'].output
    layer_output_layer_13 = layer_dict['block5_pool'].output

    layer_model_3 = Model(inputs=model.input,
                          outputs=layer_output_layer_3).predict(image)
    layer_model_13 = Model(inputs=model.input,
                           outputs=layer_output_layer_13).predict(image)

    plt.clf()
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.imshow(layer_model_3[0, :, :][:, :, i * 4])
    plt.savefig('vgg_images/' + pic_name + '_block1_pool' + '.png')

    plt.clf()
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.imshow(layer_model_13[0, :, :][:, :, i * 32])
    plt.savefig('vgg_images/' + pic_name + '_block5_pool' + '.png')
