import modules

config = {
    "type": "modules.erfnet.Encoder"
}
# build an encoder
encoder = modules.model_factory.buildModel(config)
print(encoder)

config = modules.model_factory.completeConfig(config)
print('Config after filling in defaults')
print(config)

config = {
    "type": "modules.erfnet.Encoder",
    'args': {
        'activation': {
            'type': 'LeakyReLU',
            'negative_slope': 0.1
        },
        
        'filter_channels': 32
    }
}
encoder = modules.model_factory.buildModel(config)

print(encoder)


# this tests the advanced features when creating a multi task network
import yaml
config = yaml.safe_load(open('config.yaml', 'rt'))
model = modules.model_factory.buildModel(config)

