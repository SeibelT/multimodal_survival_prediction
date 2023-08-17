from torchvision.transforms import RandomHorizontalFlip,RandomVerticalFlip
from torchvision import datasets, transforms
import torch
from ffcv_pl.ffcv_utils.utils import FFCVPipelineManager
from ffcv.fields.decoders import IntDecoder,FloatDecoder,NDArrayDecoder
from ffcv.fields.rgb_image import RandomResizedCropRGBImageDecoder, CenterCropRGBImageDecoder,SimpleRGBImageDecoder
from ffcv.loader import OrderOption
from ffcv.transforms import ToTensor, ToTorchImage
from ffcv_pl.data_loading import FFCVDataModule
from ffcv_pl.ffcv_utils.augmentations import DivideImage255

def ffcvmodule(batch_size,num_workers,is_dist,**kwargs):
        train_manager = getFFCVPipelineManager("train") 
        test_manager = getFFCVPipelineManager("test")
        val_manager = getFFCVPipelineManager("val") 
        return FFCVDataModule(batch_size=batch_size, num_workers=num_workers, train_manager=train_manager, val_manager=val_manager,test_manager=test_manager,
                                 is_dist=is_dist)

def getFFCVPipelineManager(mode):
    pipeline_transforms=[

                    # tile pipeline
                    [SimpleRGBImageDecoder(),
                        ToTensor(),
                        ToTorchImage(),
                        DivideImage255(dtype=torch.float32),
                        RandomHorizontalFlip(p=0.5),RandomVerticalFlip(p=0.5)],
                    
                    # genomics pipeline
                    [NDArrayDecoder(),
                        ToTensor()
                        ],
                    
                    # censorship pipeline
                    [IntDecoder(),
                        ToTensor()
                        ],
                    
                    # label pipeline
                    [IntDecoder(),
                        ToTensor()
                        ],
                    
                    # label cont pipeline
                    [FloatDecoder(),
                        ToTensor()
                        ],
                    
                    
                ]
    if mode=="train":
        return FFCVPipelineManager("/globalwork/seibel/beton_ds/tile_survival_train.beton",
                                   pipeline_transforms=pipeline_transforms,
                                   ordering=OrderOption.RANDOM)  # random ordering for training

    if mode=="test":
        return FFCVPipelineManager("/globalwork/seibel/beton_ds/tile_survival_test.beton",  # previously defined using dataset_creation.py
                                   pipeline_transforms=pipeline_transforms,
                                   ordering=OrderOption.RANDOM)  # random ordering for training
    if mode=="val":
        return FFCVPipelineManager("/globalwork/seibel/beton_ds/tile_survival_val.beton",  # previously defined using dataset_creation.py
                                   pipeline_transforms=pipeline_transforms,
                                   ordering=OrderOption.RANDOM)  # random ordering for training
