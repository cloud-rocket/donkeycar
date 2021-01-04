import tensorflow.compat.v1 as tf
import uff
import tensorrt as trt 
tf.disable_v2_behavior()

def frozen_pb_to_plan(model_path, 
                      output_path,
                      tensor_in_name,
                      tensor_out_name, 
                      input_size,
                      data_type=trt.float32,
                      max_batch_size=1,
                      max_workspace=1<<30,
                      tensorboard_dir=None):

    # infer with pb model
    graph_def = tf.GraphDef()
    with tf.io.gfile.GFile(model_path, "rb") as f:
        graph_def.ParseFromString(f.read())
    
    # convert TF frozen graph to uff model
    uff_model = uff.from_tensorflow_frozen_model(model_path, [tensor_out_name])
    
    # create uff parser
    parser = trt.UffParser()
    parser.register_input(tensor_in_name, input_size)
    parser.register_output(tensor_out_name)

    # create trt logger and builder
    trt_logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(trt_logger)
    builder.max_batch_size = max_batch_size
    builder.max_workspace_size = max_workspace
    builder.fp16_mode = (data_type == trt.float16)

    # parse the uff model to trt builder
    network = builder.create_network()
    parser.parse_buffer(uff_model, network)

    # build optimized inference engine
    engine = builder.build_cuda_engine(network)

    # save inference engine
    with open(output_path, "wb") as f:
        f.write(engine.serialize())
        
        
        
BATCH_SIZE = 1
H, W, C = 299, 299, 3

if __name__ == "__main__":
    '''
    generate the inference engine 
    '''
    pb_model_path = "/home/meir/mycar/models/linear.pb"
    plan_model_path = "/home/meir/mycar/models/linear.plan"
    input_node_name = "img_in"
    output_node_name = "n_outputs0/BiasAdd"

    frozen_pb_to_plan(pb_model_path,
                      plan_model_path,
                      input_node_name,
                      output_node_name,
                      [C, H, W],
                      data_type=trt.float32, # change this for different TRT precision
                      max_batch_size=1,
                      max_workspace=1<<30)
        