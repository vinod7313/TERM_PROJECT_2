#Import Modules

import numpy as np    #to handle array and matrix dataset
import os    #to create folder/directories in the system
import six.moves.urllib as urllib    #to open URL
import tarfile   #to open model (.tar file) 
import tensorflow as tf   #for numerical computation

from object_detection.utils import label_map_util   #utilities for object detection i.e label
from PIL import Image           #image control/open image
from IPython.display import display    #to display output of object dection
from object_detection.utils import visualization_utils as vis_util    #to visualize array image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# # Model preparation 
# Which model to download ?
MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'          #save in variables

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join(r'C:\Users\hp\Desktop\IIT Kharagpur Notes\Semester I\Seminar-I\My\Dataset\Fruits', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90  #diveded into no. of classes

#Download Model from URL & if model already exist it will show model already exists

if not os.path.exists(MODEL_NAME + '/frozen_inference_graph.pb'):
	print ('Downloading the model')
	opener = urllib.request.URLopener()
	opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
	tar_file = tarfile.open(MODEL_FILE)
	for file in tar_file.getmembers():
	  file_name = os.path.basename(file.name)
	  if 'frozen_inference_graph.pb' in file_name:
	    tar_file.extract(file, os.getcwd())
	print ('Download complete')
else:
	print ('Model already exists')


#Load/Save a (frozen) Tensorflow model into memory
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.compat.v1.GraphDef()
  with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


#Loading label map
# Label mapping i.e. it map indices to category names
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

#it puts input image here and convert it into array i.e. coverts image into numpy array i.e. no.s

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


#Detection 
# Path of input image
PATH_TO_TEST_IMAGES_DIR = 'C:/Users/hp/Desktop/IIT Kharagpur Notes/Semester I/Seminar-I/My/Dataset/Fruits/1. Train Images/'


# Pick images individually i.e. one by one and put it here and save it in variables
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 122) ]

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)


#Detects the imported images by downloading the model and labelling the images using that model
with detection_graph.as_default():
  with  tf.compat.v1.Session(graph=detection_graph) as sess:
    for image_path in TEST_IMAGE_PATHS:
      image = Image.open(image_path) # opens the image path and saves the image in the variables
      #converts the image into numpy array (red, green, blue)
      image_np = load_image_into_numpy_array(image) 
      #detect the converted numpy array
      image_np_expanded = np.expand_dims(image_np, axis=0) 
      image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
      #bounding detected object in the images by boxes
      boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
      #shows the %detection accuracy
      scores = detection_graph.get_tensor_by_name('detection_scores:0')
      #shows the name of object
      classes = detection_graph.get_tensor_by_name('detection_classes:0')
      num_detections = detection_graph.get_tensor_by_name('num_detections:0')   
     
      #actual detection.
      #makes the dictionary of boxes, score, class of object, and no. detection
      (boxes, scores, classes, num_detections) = sess.run(
          [boxes, scores, classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})
     
      #visualization of the results of detection by visualization utilities
      vis_util.visualize_boxes_and_labels_on_image_array(image_np,np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          use_normalized_coordinates=True,
          line_thickness=8)     
          
      #used Image module to display the array image
      im = Image.fromarray(image_np)  
      display(im)
      p_list = image_path.split('/')
      
      #to save the output images of object detection into a particular folders
      im.save('C:/Users/hp/Desktop/IIT Kharagpur Notes/Semester I/Seminar-I/My/Dataset/Fruits/2. Output/{}'.format(p_list[11]))