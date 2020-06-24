from tensorflow.keras import Model
import numpy as np

def get_intermediate_layer(model, layer_name, test_input):
    return Model(inputs=model.input, outputs=model.get_layer(layer_name).output)

def get_intermediate_output(model, layer_name, test_input):
    intermediate_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    return intermediate_model(test_input)


def get_output_of_all_layers(model, test_input):
    output_of_all_layers = dict()
    for count, layer in enumerate(model.layers):
        layer_name = layer.name
        #skip the input layer
        if count == 0:
            continue
        output_of_all_layers[layer_name] = np.array(get_intermediate_output(model, layer_name, test_input))
    
    return output_of_all_layers