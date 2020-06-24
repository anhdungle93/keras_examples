from tensorflow.keras import Model

def get_intermediate_layer(model, layer_name, test_input):
    return Model(inputs=model.input, outputs=model.get_layer(layer_name).output)

def get_intermediate_output(model, layer_name, test_input):
    intermediate_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    return intermediate_model(test_input)


def get_output_of_all_layers(model, test_input):
    output_of_all_layers = []
    for count, layer in enumerate(model.layers):

        #skip the input layer
        if count == 0:
            continue
        output_of_all_layers.append(get_intermediate_output(model, layer, test_input))
    
    return output_of_all_layers