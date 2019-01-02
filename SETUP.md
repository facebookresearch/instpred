### Recommended environment 

This code was run in the following environment:
- NVIDIA GPUs, Linux, Python 2 and Anaconda/Miniconda 2
- CUDNN 7.0 and CUDA 9.0

We provide below the corresponding installation instructions.

### Installation

1/ Create a new conda environment and activate it:

```
CONDA_ENV=instpred
conda create --name $CONDA_ENV pip
source activate $CONDA_ENV
```

2/ Install Pytorch including Caffe2 and the detectron module, by following the instructions on the [Caffe2 website](https://caffe2.ai/docs/getting-started.html). Also install the packages for running the tutorials listed [here](https://caffe2.ai/docs/tutorials).
In our case, we ran the following commands **only adapted if you are in our recommended environment**:
```
pip install future protobuf hypothesis
conda install pytorch-nightly -c pytorch
```

Now test that your Caffe2 version has GPU support and the detectron module:
```
# Check that Caffe2 build was successful
python2 -c 'from caffe2.python import core' 2>/dev/null && echo "Success" || echo "Failure"

# Check that Caffe2 GPU build was successful
# This must print a number > 0 in order to use Detectron
python2 -c 'from caffe2.python import workspace; print(workspace.NumCudaDevices())'

# test Detectron module: should print OK
CAFFE2_PATH=/path_to_your_conda_root/envs/$CONDA_ENV/lib/python2.7/site-packages/caffe2/ # USERTODO adapt
python $CAFFE2_PATH/python/operator_test/roi_align_rotated_op_test.py 
```
<!-- TODO : check that the last test is at all necessary and check difference with below test -->

3/ Install COCO API. This requires cython and matplotlib packages:
```
pip install Cython matplotlib
```
Next, you can clone the github repo for this package and install the Python API:
```
COCOAPI=/desired/path/to/cocoapi # USERTODO adapt
git clone https://github.com/cocodataset/cocoapi.git $COCOAPI
cd $COCOAPI/PythonAPI
# Install
python2 setup.py install --user
```
<!-- TODO : check this -->

4/ Follow the remaining instructions to install Detectron [here](https://github.com/facebookresearch/Detectron/blob/master/INSTALL.md) (only those necessary for inference), i.e.
```
DETECTRON=/desired/path/to/detectron  # USERTODO adapt
git clone https://github.com/facebookresearch/Detectron.git $DETECTRON
pip install -r $DETECTRON/requirements.txt
cd $DETECTRON && make
python $DETECTRON/detectron/tests/test_spatial_narrow_as_op.py
```

5/ Finally, to reproduce our results, you need the Cityscapes dataset. 
- Download the *leftImg8bit_sequence_trainvaltest* and *gtFine_trainvaltest* packages from the [Cityscapes dataset website](https://www.cityscapes-dataset.com/downloads/), and into your chosen CITYSCAPES_IMAGES and CITYSCAPES_GT_ROOT directories (necessary for the Cityscapes evaluation script).
- Download the Cityscapes annotations converted to COCO json format from [coco_style_annotations](coco_style_annotations) (necessary for the COCO style dataset loader) and into your chosen CITYSCAPES_JSON_ANNOTATIONS directory.
- Download and install the Cityscapes scripts [here](https://github.com/mcordts/cityscapesScripts)

```
CITYSCAPES_SCRIPTS=/desired/path/to/cityscapes_scripts # USERTODO adapt
git clone https://github.com/mcordts/cityscapesScripts.git $CITYSCAPES_SCRIPTS

```
Note: you can specify a more appropriate location for where the Cityscapes evaluation script will log its results. 
To do so, make the following modification in $CITYSCAPES_SCRIPTS/cityscapesscripts/evaluation/evalInstanceLevelSemanticLabeling.py:
```
--- a/cityscapesscripts/evaluation/evalInstanceLevelSemanticLabeling.py
+++ b/cityscapesscripts/evaluation/evalInstanceLevelSemanticLabeling.py
@@ -120,8 +120,13 @@ if 'CITYSCAPES_DATASET' in os.environ:
 else:
     args.cityscapesPath = os.path.join(os.path.dirname(os.path.realpath(__file__)),'..','..')
 
+if 'OUTPUTS_DIR' in os.environ:
+    args.outputsPath = os.environ['OUTPUTS_DIR']
+else:
+    args.outputsPath = args.cityscapesPath
+
 # Parameters that should be modified by user
-args.exportFile         = os.path.join( args.cityscapesPath , "evaluationResults" , "resultInstanceLevelSemanticLabeling.json" )
+args.exportFile         = os.path.join( args.outputsPath , "evaluationResults" , "resultInstanceLevelSemanticLabeling.json" )
 args.groundTruthSearch  = os.path.join( args.cityscapesPath , "gtFine" , "val" , "*", "*_gtFine_instanceIds.png" )
 ```
 
Then, you can proceed with installing the package, with:

```
cd $CITYSCAPES_SCRIPTS
pip install .
```

- Add the following to your Detectron dataset catalog (to be found in $DETECTRON/detectron/datasets/dataset_catalog.py:
```
    'cityscapes_fine_instanceonly_seg_sequences_train': {
        _IM_DIR:
            _DATA_DIR + '/cityscapes_sequences/images',
        _ANN_FN:
            _DATA_DIR + '/cityscapes_sequences/annotations/instancesonly_gtFine_train.json',
        _RAW_DIR:
            _DATA_DIR + '/cityscapes_sequences/raw'
    },
    'cityscapes_fine_instanceonly_seg_sequences_val': {
        _IM_DIR:
            _DATA_DIR + '/cityscapes_sequences/images',
        _ANN_FN:
            _DATA_DIR + '/cityscapes_sequences/annotations/instancesonly_gtFine_val.json',
        _RAW_DIR:
            _DATA_DIR + '/cityscapes_sequences/raw'
    },
    'cityscapes_fine_instanceonly_seg_sequences_test': {
        _IM_DIR:
            _DATA_DIR + '/cityscapes_sequences/images',
        _ANN_FN:
            _DATA_DIR + '/cityscapes_sequences/annotations/instancesonly_gtFine_test.json',
        _RAW_DIR:
            _DATA_DIR + '/cityscapes_sequences/raw'
    },
   ```
- Check the structure of the dataset directory that is expected by Detectron [here](https://github.com/facebookresearch/Detectron/blob/master/detectron/datasets/data/README.md): in particular the image directory should be adapted. Then, you can create symlinks to the adequate destinations for images, json annotations and raw annotations.
```
mkdir $DETECTRON/detectron/datasets/data/cityscapes_sequences
ln -s $CITYSCAPES_IMAGES $DETECTRON/detectron/datasets/data/cityscapes_sequences/images
ln -s $CITYSCAPES_JSON_ANNOTATIONS $DETECTRON/detectron/datasets/data/cityscapes_sequences/annotations
ln -s $CITYSCAPES_GT_ROOT $DETECTRON/detectron/datasets/data/cityscapes_sequences/raw
```
<!-- todo : provide the json annotations -->

6/ Finally install the following remaining necessary packages:
```
conda install opencv
pip install visdom pyyaml torchnet
```
<!-- todo check what we can get rid of -->

7/ **Important**: To obtain the same numerical results as us with the provided models, you must make the following changes in you Detectron library, i.e. in $DETECTRON/detectron/utils/boxes.py:
```
--- a/detectron/utils/boxes.py
+++ b/detectron/utils/boxes.py
@@ -183,9 +183,9 @@ def bbox_transform(boxes, deltas, weights=(1.0, 1.0, 1.0, 1.0)):
     # y1
     pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
     # x2 (note: "- 1" is correct; don't be fooled by the asymmetry)
-    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w - 1
+    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
     # y2 (note: "- 1" is correct; don't be fooled by the asymmetry)
-    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h - 1
+    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h
 
     return pred_boxes
```
This reverts the correction of a minor bug inside Detectron's bbox_transform and allows our provided Mask R-CNN model to run in the same setting it was trained in (i.e. before the bug was corrected).
**If instead, you want to train a F2F model from scratch, using published Detectron models or your own Detectron models, you should not proceed to reverting this.**


<!-- todo : finish this -->

### Bonus
#### Paths

Adapt the paths in [environment_configuration.py](environment_configuration.py) to your setup.

#### Signal handling for automatic job relaunching when the job receives a SIGUSR1 signal
To adapt this to your environment :
- look for USERTODO in autoregressive_training.py
- right below, fill in ```command =...``` to whatever command should be launched
in your environment to relaunch the job when it receives the signal.

Note: this is only useful if you are in an environment where your jobs can get
interrupted by SIGUSR1 signals. Otherwise you can ignore these instructions.

<!-- FINALTODO make sure we should not entirely take this out -->
