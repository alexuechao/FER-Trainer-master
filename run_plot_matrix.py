import os, sys
if __name__ == '__main__':
  cmd = 'python ./plot_rgb_confusion_matrix.py \
                    --model_path ./models_id/Alldb_sgd_0.001_MultiStepLR_rgb_64_expand_id031_centercrop_resnet_cut/Val_model.t7 \
                    --split Valing \
                    --input_shape 64'
  print(cmd)
  os.system('{}'.format(cmd))
  #os.system('nohup {}.log 2>&1 &'.format(cmd))