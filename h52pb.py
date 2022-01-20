
from tensorflow.keras.models import load_model, save_model


if __name__ == '__main__':
    model = load_model('./mask_detector.model')
    save_model(model, './mask_detector')
    model.summary()
    print(model.layers[0].name, model.layers[0].get_input_shape_at(0))
    print(model.layers[-1].name, model.layers[-1].get_output_at(0).name)
