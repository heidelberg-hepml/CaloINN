

_photon_detector_spec = {
    "num_layers": 5,
    "layers": {
        "layer0": {
            "num_angular": 1,
            "num_radial": 8, 
            "radial_edges": [0, 5, 10, 30, 50, 100, 200, 400, 600]
        },  
        "layer1": {
            "num_angular": 10,
            "num_radial": 16, 
            "radial_edges": [0, 2, 4, 6, 8, 10, 12, 15, 20, 30, 40, 50, 70, 90, 120, 150, 200]
        },  
        "layer2": {
            "num_angular": 10,
            "num_radial": 19, 
            "radial_edges": [0,2,5,10,15,20,25,30,40,50,60,80,100,130,160,200,250,300,350,400] 
        },  
        "layer3": {
            "num_angular": 1,
            "num_radial": 5, 
            "radial_edges": [0,50,100,200,400,600]
        },  
        "layer4": {
            "num_angular": 1,
            "num_radial": 5, 
            "radial_edges": [0,100,200,400,1000,2000]
        },  
    }
}

_pion_detector_spec = {
    "metadata": {
        "num_layers": 7
    },
    "layers": {
        "layer0": {
            "num_angular": 1,
            "num_radial": 8,
            "radial_edges": [0,5,10,30,50,100,200,400,600],
        },
        "layer1": {
            "num_angular": 10,
            "num_radial": 10,
            "radial_edges": [0,1,4,7,10,15,30,50,90,150,200],
        },
        "layer2": {
            "num_angular": 10,
            "num_radial": 10,
            "radial_edges": [0,5,10,20,30,50,80,130,200,300,400],
        },
        "layer3": {
            "num_angular": 1,
            "num_radial": 5,
            "radial_edges": [0,50,100,200,400,600],
        },
        "layer4": {
            "num_angular": 10,
            "num_radial": 15,
            "radial_edges": [0,10,20,30,50,80,100,130,160,200,250,300,350,400,1000,2000],
        },
        "layer5": {
            "num_angular": 10,
            "num_radial": 16,
            "radial_edges": [0,10,20,30,50,80,100,130,160,200,250,300,350,400,600,1000,2000],
        },
        "layer6": {
            "num_angular": 1,
            "num_radial": 10,
            "radial_edges": [0,50,100,150,200,250,300,400,600,1000,2000],
        },
    }
}

## Photon Metadata Sanitycheck
#total1 = 0; 
#total2 = 0; 
#for layer_key in _photon_detector_spec["layers"]: 
#    layer = _photon_detector_spec["layers"][layer_key]; 
#    total1 += layer["num_angular"] * layer["num_radial"]; 
#    total2 += layer["num_angular"] * (len(layer["radial_edges"]) - 1)
#print(total1); 
#print(total2); 

## Pion Metadata Sanitycheck
#total1 = 0; 
#total2 = 0; 
#for layer_key in _pion_detector_spec["layers"]: 
#    layer = _pion_detector_spec["layers"][layer_key]; 
#    total1 += layer["num_angular"] * layer["num_radial"]; 
#    total2 += layer["num_angular"] * (len(layer["radial_edges"]) - 1)
#print(total1); 
#print(total2); 








