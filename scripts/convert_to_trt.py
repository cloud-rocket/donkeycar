from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tensorflow.python.framework import graph_io
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


model_path = "/home/meir/mycar/models/linear.pb"
output_path = "/home/meir/mycar/models"


output_names = ['n_outputs0/BiasAdd', 'n_outputs1/BiasAdd']

# infer with pb model
frozen_graph = tf.GraphDef()
with tf.io.gfile.GFile(model_path, "rb") as f:
    frozen_graph.ParseFromString(f.read())
    


    trt_graph = trt.create_inference_graph(
      input_graph_def=frozen_graph,
      outputs=output_names,
      max_batch_size=1,
      max_workspace_size_bytes=1 << 25,
      precision_mode='FP16',
      minimum_segment_size=50
    )
    
    
    graph_io.write_graph(trt_graph, output_path, 'trt_linear.pb', as_text=False)