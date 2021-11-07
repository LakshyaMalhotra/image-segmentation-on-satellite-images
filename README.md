# Indirect Models
Detecting water using other indirect features.

For more information, please check each model documentation and READM.md.
## Content
- [Overview](#overview)
- [General Approach](#general-approach)
    - [Main Broad Categories](#main-board-categories)
    - [Vegetation approaches](#vegetation-approach)
    - [LandCoverNet models](#landcovernet-models)
    - [Topographical Models](#topographical-models)
    - [Chip models](#chip-models)

## Overview

## General Approach

We want to focus on approaches that do not try to directly detect water from the satellite images, but rather infer water presence based on other factors.
This is particularly important when considering small water bodies and vegetated water bodies.


### Main Broad Categories
- Proxy variables. And later ensembling
- Generating labels through proxy variables
- Terrain patch characterization
- Classifying land cover based on similarity

![General Workflow](https://github.com/OmdenaAI/zzapp/blob/master/indirect-models/images/workflow.png)

### Vegetation approaches
Created training data for vegetated water bodies on the basis of :
- vv/vh ratio and temporal analysis of global water occurrence 
- Generated features and uploaded to s3
- Calculated terrain derivatives on the HAND data

![vegetation](https://github.com/OmdenaAI/zzapp/blob/master/indirect-models/images/vegetation.png)

### LandCoverNet models
Retrained the model on the bigger landcovernet dataset and generated features based on the land cover classes 
Mapped to Amhara and Obuasi GeoTIFF files
Obtained land cover features are in the form of percent of a given land cover class on MGRS chunks.

Actual Image   |   Original Mask   |     Model Output 
:----------: |    :-----------:  |      :----------: 
| ![Image](https://github.com/OmdenaAI/zzapp/blob/master/indirect-models/landcovernet_model/images/29PKL_08_20181011_RGB.jpg) |  ![Mask](https://github.com/OmdenaAI/zzapp/blob/master/indirect-models/landcovernet_model/images/29PKL_08_20181011_mask.jpg) |  ![Output](https://github.com/OmdenaAI/zzapp/blob/master/indirect-models/landcovernet_model/images/29PKL_08_20181011_output_mask.jpg) |

### Topographical Models
Features of interest: convergence_index_mean, twi_mean, ls_factor_max, channel_network_distance_mean, tpi500_mean, rel_slope_position_mean
Created notebooks and documentation for the description of the features generated
Trained on Obuasi west, validated on Obuasi east
Figure: Predictions on Obuasi east data

  Obuasi Actual Positive/Negative Chunks   |       |     Obuasi Predicted Positive/Negative Chunks
:----------: |    :--------------:  |      :----------: 
| ![Obuasi Actual](https://github.com/OmdenaAI/zzapp/blob/master/indirect-models/Topographic%20models%20and%20features/images/water_labels.png) |      |  ![Obuasi Predicted](https://github.com/OmdenaAI/zzapp/blob/master/indirect-models/Topographic%20models%20and%20features/images/predictions.png) |

### Chip models
Performance using east/west separation greatly dropped, indicating that the features were previously too correlated and that the model was not separating between water/non-water
AUC almost 0.5 in val, or worse, indicating that the model ends up predicting the opposite of what it should
There's some stability issues we have to still check.

![Chip Model Performance](https://github.com/OmdenaAI/zzapp/blob/master/indirect-models/images/chips.png)
