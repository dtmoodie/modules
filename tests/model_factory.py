
import modules

config = {
    "type": "ErfnetEncoder"
}
# build an encoder
encoder = modules.model_factory.buildModel(config)
print(encoder)
config = modules.model_factory.completeConfig(config)
print(config)

config = {
    "type": "ErfnetEncoder",
    'args': {
        'activation': {
            'type': 'LeakyReLU'
        },
        'negative_slope': 0.1,
        'filter_channels': 32
    }
}
encoder = modules.model_factory.buildModel(config)

print(encoder)
