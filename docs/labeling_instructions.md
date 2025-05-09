# SKAI Example Labeling Instructions

Updated: Oct 23, 2024

## Overview

The goal of this labeling task is to provide the SKAI machine learning model
with example images of damaged and undamaged buildings, so that it can learn to
identify buildings damaged in a disaster.

## Instructions

You will be presented with pairs of ‘BEFORE'- and ‘AFTER'- satellite images of
individual buildings, along with the surrounding environment for context. Focus
on the building marked by the red frame in the center of the image. Your task is
to compare the ‘BEFORE' and ‘AFTER' images and determine whether the building
has been damaged as a result of the disaster.

![image](images/example_image.png)

Sometimes it is difficult to tell if a building is damaged or not. Please do
your best in these cases, and remember that a small number of mistakes will not
mislead the model very much. If there are multiple buildings in the frame,
evaluate only the building in the very center. It can also be helpful to check
the surroundings of the building for clues: if all the nearby buildings are
clearly damaged, or if there is rubble and debris everywhere, it's likely that
the building is damaged.

## User Interface

This is a screenshot of the Eagle Eye labeling tool in progress. It includes the
BEFORE (Pre-Event) and AFTER (Post-Event) images along with the choices for
labels on the bottom.

![eagle_eye_labeling_interface](images/eagle_eye_labeling_interface.png)

-  For each example, please choose the best label, and then click the submit
   button.

## Labels explanations and examples

Please label each building from the following choices:

### no_damage

The building was not damaged by the disaster event. This includes cases where
the building was already damaged prior to the disaster (i.e. it is clearly
damaged in the ‘BEFORE' (left) image).

![no_damage](images/labeling_example_no_damage.png)

### minor_damage

The building has minor damage such as: partially burnt, water surrounding
structure, volcanic flow nearby, roof elements missing or visible cracks.

![minor_damage](images/labeling_example_minor.png)

### major_damage

The building has major damage such as: partial wall or roof collapse,
encroaching volcanic flow or surrounded by water/mud.

![major_damage](images/labeling_example_major.png)

### destroyed

The building was effectively destroyed such as: the building is burned to the
ground, completely collapsed, partially/completely covered with water/mud, and
became rubble.

![destroyed](images/labeling_example_destroyed.png)

### bad\_example

Bad example labels cover many cases in which you cannot provide relevant
labels. Some concrete examples include:

1. **No building** in the center of the red box (due to a mistake
    in our building detection). Examples include cars, patches of forest,
    shipping containers, etc.

    ![image](https://storage.googleapis.com/skai-public/documentation_images/labeling_instructions/no_building_example.png)

2. **Unclear images** - The building in either BEFORE and/or AFTER
    disaster image cannot be seen. For example when it is covered by
    clouds, or is too blurry.

    ![image](https://storage.googleapis.com/skai-public/documentation_images/labeling_instructions/after_image_unclear_example.png)

3. **Extreme misalignment** - The images are so misaligned that the
    correct building is mostly or completely out of the red frame. Don't
    use this label if the correct building is still mostly in the red frame.

    ![image](https://storage.googleapis.com/skai-public/documentation_images/labeling_instructions/extreme_misalignment_example.png)

## Challenging cases

**‘BEFORE'/‘AFTER' images misaligned**
The ‘BEFORE' and ‘AFTER' images are misaligned, and as a result, the building in
the ‘AFTER' image is not in the center of the red frame.

![image](https://storage.googleapis.com/skai-public/documentation_images/labeling_instructions/misaligned_example.png)

**Recommendation:** Please try your best to identify the building in the ‘AFTER'
image that corresponds to the center building in the ‘BEFORE' image, and decide
whether that building is damaged, regardless of alignment.

**Multiple buildings in frame**
There are two buildings in the center red frame.

![image](https://storage.googleapis.com/skai-public/documentation_images/labeling_instructions/multiple_buildings_example.png)

**Recommendation:** Please focus on the building in the center of the frame,
even if there are off-center buildings that are larger.
