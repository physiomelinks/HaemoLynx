import sys
import re

with open("examples/carotid_image_to_model.py", "r") as f:
    content = f.read()

# 1. Remove the hard firebreak
content = re.sub(r'    if entropy_map is not None and pre_config_dict\.get\("enable_shannon_entropy", True\):\n        threshold = pre_config_dict\.get\("shannon_entropy_threshold", 0\.95\)\n        uncertain_mask = entropy_map > threshold\n        image\[uncertain_mask\] = 0\.0\n\n', '', content)

# 2. Update hysteresis
old_hysteresis = """    if pre_config_dict.get("enable_hysteresis_threshold", True):
        binary = preprocessing.hysteresis_threshold(
            image,
            low=pre_config_dict.get("hysteresis_threshold_low", 0.2),
            high=pre_config_dict.get("hysteresis_threshold_high", 0.4)
        )
    else:
        from skimage.filters import threshold_otsu
        binary = image > threshold_otsu(image)"""

new_hysteresis = """    if pre_config_dict.get("enable_hysteresis_threshold", True):
        if entropy_map is not None and pre_config_dict.get("enable_shannon_entropy", True):
            from skimage.morphology import reconstruction
            
            # Phase 2.5: Coupled Probability-Entropy Hysteresis
            core_mask = (image > pre_config_dict.get("hysteresis_threshold_high", 0.4)) & \\
                        (entropy_map < pre_config_dict.get("shannon_core_max", 0.5))
                        
            transition_mask = (image > pre_config_dict.get("hysteresis_threshold_low", 0.2)) & \\
                              (entropy_map <= pre_config_dict.get("shannon_transition_max", 0.95))
                              
            allowed_mask = core_mask | transition_mask
            
            # Morphological reconstruction acts as the dual-parameter hysteresis
            binary = reconstruction(seed=core_mask, mask=allowed_mask, method='dilation').astype(bool)
        else:
            binary = preprocessing.hysteresis_threshold(
                image,
                low=pre_config_dict.get("hysteresis_threshold_low", 0.2),
                high=pre_config_dict.get("hysteresis_threshold_high", 0.4)
            )
    else:
        from skimage.filters import threshold_otsu
        binary = image > threshold_otsu(image)"""

content = content.replace(old_hysteresis, new_hysteresis)

with open("examples/carotid_image_to_model.py", "w") as f:
    f.write(content)
