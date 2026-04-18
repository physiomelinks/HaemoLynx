# ImageLynx Pipeline TODOs and Notes

## General Data Analysis
- Ability to compare datasets - Dave's suggestion
- Summarise by BO (Branch Order) in statistics
- Resistance should be from start of arteriole to end of venule
- Mean distance of object (classifier) to each capillary type and BO
- Overall list of every vessel and its properties

## Haemodynamics & Resistance
- **HD note:** Eventually add script to run resistance measurements between every BO1 (arteriole) and every (non-arteriole) capillary node, and between every node.
- Automate the selection of resistance node pairs (e.g. `RESISTANCE_NODE_PAIR = (426, 509)`)
- Diameters etc. should be automated.
- **HD note:** There should be a manual option to add in in-vivo diameters, and an option to read in diameters from the original image (via FWHM).
- **HD note:** This no longer features the ability to manually define a limited number of user-determined vessels (i.e. endoneurial vessels), which can't be done automatically. Not relevant for Alice but relevant generally.

## Future Testing
- Run through current image-to-model functionality with CB binary-mask and add fixes/features on the fly.
