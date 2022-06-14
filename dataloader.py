from skimage import io
import numpy as np

def gen_data(patch_size):
  hs = io.imread('./dataset/2013_IEEE_GRSS_DF_Contest_CASI.tif')
  lid = io.imread('./dataset/2013_IEEE_GRSS_DF_Contest_LiDAR.tif')
  lid = np.expand_dims(lid,-1)

  net = np.concatenate([hs,lid],-1)
  for i in range(net.shape[-1]):
    net[:,:,i] = net[:,:,i]/np.amax(net[:,:,i])


  training = np.loadtxt('dataset/train.txt')
  testing = np.loadtxt('dataset/test.txt')

  # patch_size = 7

  full = np.zeros((net.shape[0]+patch_size//2, net.shape[1]+patch_size//2, net.shape[2]))
  full[patch_size//2:patch_size//2 +net.shape[0], patch_size//2 :patch_size//2 +net.shape[1], :] = net


  xy_training = training[:,[1,0]].astype('int')
  xy_testing = testing[:,[1,0]].astype('int')

  y_train = training[:,[2]].astype('int') -1 
  y_test = testing[:,[2]].astype('int') -1 

  training_patch = [ full[element[0]:element[0]+patch_size, element[1]:element[1]+patch_size, :] for element in xy_training]
  training_patch = np.asarray(training_patch)
  testing_patch = [ full[element[0]:element[0]+patch_size, element[1]:element[1]+patch_size, :] for element in xy_testing]
  testing_patch = np.asarray(testing_patch)

  return training_patch, y_train, testing_patch, y_test

if __name__ == '__main__':
  gen_data(patch_size)