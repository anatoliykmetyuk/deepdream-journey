"""
Some front-end methods for deepdream.
"""

import os

from deepdream import *

def load_img(name, base='.', scale=1):
  """
  Loads an image and converts it to a float32 numpy array.
  """
  img = PIL.Image.open(os.path.join(base, name))
  width, height = img.size
  img = img.resize((int(round(width * scale)), int(round(height * scale))))
  return np.float32(img)

def dream_layers(net, img, out_dir, **dreamparams):
  """
  Launches deepdream on all the layers of the network.
  Outputs the results to the out_dir.
  """
  for frame_id, target_layer in enumerate(net.blobs.keys()):
    try:
      frame = deepdream(net, img, end=target_layer, **dreamparams)
      filename = os.path.join(out_dir, "%i_%s.jpg" % (frame_id, target_layer.replace('/', ':')))
      PIL.Image.fromarray(np.uint8(frame)).save(filename)
    except ValueError: pass

def dream_deep(net, img, out_dir, frames_n, scale = 0.05, dh = 0.5, dw = 0.5, start_idx = 0, **dreamparams):
  """
  Feeds the output of deepdream back to deepdream in a loop.
  Produces frames_n frames, saves them to out_dir.
  """
  frame   = img
  frame_i = start_idx

  h, w = frame.shape[:2]
  s = scale # scale coefficient
  for i in xrange(frames_n):
    frame = deepdream(net, frame, **dreamparams)
    filename = os.path.join(out_dir, "%06d.jpg" % frame_i)
    PIL.Image.fromarray(np.uint8(frame)).save(filename)
    frame = nd.affine_transform(frame, [1-s,1-s,1], [h*s*(dh+s),w*s*(dw+s),0], order=1)
    frame_i += 1