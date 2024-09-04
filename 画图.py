import matplotlib.pyplot as plt

def visualize_conv3d_filters(model, layer_name, num_filters=5):
    # 获取指定层的卷积核
    conv_layer = dict(model.named_modules())[layer_name]
    weights = conv_layer.weight.data.cpu().numpy()

    fig, axs = plt.subplots(num_filters, weights.shape[2], figsize=(15, num_filters * 3))
    
    for i in range(num_filters):
        for j in range(weights.shape[2]):
            axs[i, j].imshow(weights[i, 0, j, :, :], cmap='gray')
            axs[i, j].axis('off')
    
    plt.show()

# 假设你有一个模型，并且模型中有一个叫 'conv3d_1' 的 3D 卷积层
visualize_conv3d_filters(model, 'conv3d_1')
