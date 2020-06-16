import os
import torch

src_weight = 'epoch_12.pth'
model = torch.load(src_weight)

weight = model['state_dict']
filter_names = []
layer_names = list(weight.keys())
for layer_name in layer_names:
    if 'conv' in layer_name and 'conv2_offset.weight' in layer_name:
        #print(layer_name)
        filter_name = layer_name.replace('conv2_offset', 'bn2')
        filter_names.append(filter_name)

for layer_name in layer_names:
    if 'bn' in layer_name and 'weight' in layer_name:
        if layer_name in filter_names:
            continue
        conv_name = layer_name.replace('bn', 'conv')
        bn_bias_name = layer_name.replace('weight', 'bias')
        bn_running_mean_name = layer_name.replace('weight', 'running_mean')
        bn_running_var_name = layer_name.replace('weight', 'running_var')
        bn_weight = weight[layer_name]
        conv_weight = weight[conv_name]
        bn_bias_weight = weight[bn_bias_name]
        bn_running_mean_weight = weight[bn_running_mean_name]
        bn_running_var_weight = weight[bn_running_var_name]
        var_sqrt = torch.sqrt(bn_running_var_weight + 1e-5)
        mean = bn_running_mean_weight
        beta = bn_weight
        gamma = bn_bias_weight
        conv_bias = mean.new_zeros(mean.shape)
        conv_weight = conv_weight * (beta / var_sqrt).reshape([conv_weight.shape[0], 1, 1, 1])
        conv_bias = (conv_bias - mean) / var_sqrt * beta + gamma
        weight[conv_name] = conv_weight
        conv_bias_name = conv_name.replace('weight', 'bias')
        weight[conv_bias_name] = conv_bias
        num_batches_tracked_name = layer_name.replace('weight', 'num_batches_tracked')
        weight.pop(layer_name)
        weight.pop(bn_bias_name)
        weight.pop(bn_running_mean_name)
        weight.pop(bn_running_var_name)
        weight.pop(num_batches_tracked_name)

    if 'downsample.1' in layer_name and 'weight' in layer_name:
        conv_name = layer_name.replace('downsample.1', 'downsample.0')
        bn_bias_name = layer_name.replace('weight', 'bias')
        bn_running_mean_name = layer_name.replace('weight', 'running_mean')
        bn_running_var_name = layer_name.replace('weight', 'running_var')
        bn_weight = weight[layer_name]
        conv_weight = weight[conv_name]
        bn_bias_weight = weight[bn_bias_name]
        bn_running_mean_weight = weight[bn_running_mean_name]
        bn_running_var_weight = weight[bn_running_var_name]
        var_sqrt = torch.sqrt(bn_running_var_weight + 1e-5)
        mean = bn_running_mean_weight
        beta = bn_weight
        gamma = bn_bias_weight
        conv_bias = mean.new_zeros(mean.shape)
        conv_weight = conv_weight * (beta / var_sqrt).reshape([conv_weight.shape[0], 1, 1, 1])
        conv_bias = (conv_bias - mean) / var_sqrt * beta + gamma
        weight[conv_name] = conv_weight
        conv_bias_name = conv_name.replace('weight', 'bias')
        weight[conv_bias_name] = conv_bias
        num_batches_tracked_name = layer_name.replace('weight', 'num_batches_tracked')
        weight.pop(layer_name)
        weight.pop(bn_bias_name)
        weight.pop(bn_running_mean_name)
        weight.pop(bn_running_var_name)
        weight.pop(num_batches_tracked_name)

for name in weight.keys():
    print(name)

new_model = dict()
new_model['meta'] = model['meta']
new_model['state_dict'] = weight
torch.save(new_model, 'merge_bn.pth')
