import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np

#This will set the image on the returned axis object. Can use the returned ax to add rectangle/text.
def set_img(im, figsize=None, ax=None):
  if not ax: 
    fig,ax = plt.subplots(figsize=figsize)

  ax.imshow(im, interpolation='nearest')
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)
  return ax

#this will take in a tensor. We'll need to convert it to numpy before we grab its values. 
def draw_rect(ax, b, color):
  xy = (b[0], b[1])
  rect = patches.Rectangle(xy=xy, width=b[2], height=b[3], fill=False, linewidth=3.0 ,color=color, edgecolor=None)
  ax.add_patch(rect)

def draw_text(ax, b, text_val, color, sz=14):
  ax.text(b[0], b[1], text_val, color='white', bbox=dict(facecolor=color, alpha=0.5), fontsize=sz)

def tensor_to_numpy(t_val):
  return t_val.numpy()

#takes NUMPY array, not tensor. Returns (x, y, width, height)
def bb_wh(bb):
  assert (type(bb) == np.ndarray),"bounding box is not of type numpy!"
  return np.array([bb[1],bb[0],bb[3]-bb[1]+1,bb[2]-bb[0]+1])

def wh_bb(wh):
  assert (type(wh) == np.ndarray),"bounding box is not of type numpy!"
  return np.array([wh[1], wh[0], wh[3]+wh[1]-1, wh[2]+wh[0]-1])




