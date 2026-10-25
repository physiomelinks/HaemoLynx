# Statistics glossary

<!-- Written by `python -m haemolynx.statistics.glossary` from
     src/haemolynx/statistics/glossary.py; edit that, not this. -->

Every row of `{stem}_statistics.csv`, `{stem}_statistics_connected.csv` (and a
perturbation's statistics CSV) and `{stem}_tissue_volume.csv`, and every column of
`{stem}_branch_statistics.csv` and `{stem}_capillary_bed_joins.csv`: what it is and
its unit. Each statistics CSV row carries its unit in the Unit column and this
definition in the Notes column. A name with `/` or `N` stands for each of the
names it lists.

## Units

| Unit | Meaning |
|---|---|
| `um` | micrometres (microns) |
| `um^2` | square micrometres |
| `um^3` | cubic micrometres |
| `um/um^3` | micrometres of vessel per cubic micrometre (length density) |
| `um^2/um^3` | square micrometres of vessel wall per cubic micrometre |
| `1/um` | per micrometre |
| `1/mm` | per millimetre of vessel |
| `1/mm^3` | per cubic millimetre |
| `1/mm^2` | per square millimetre |
| `mm/s` | millimetres per second |
| `nL/min` | nanolitres per minute |
| `m^3/s` | cubic metres per second |
| `m^3/(Pa.s)` | hydraulic conductance, cubic metres per pascal second |
| `Pa` | pascals |
| `Pa.s/m^3` | hydraulic resistance, pascal seconds per cubic metre |
| `s` | seconds |
| `fL` | femtolitres |
| `cells/s` | red cells per second |
| `cells/mm` | red cells per millimetre of vessel |
| `degrees` | degrees of angle |
| `vessels` | a number of vessels (graph edges); as a route length, the vessels along it |
| `count` | a number of things other than vessels (nodes, junctions, beds, loops, samples) |
| `fraction` | a share between 0 and 1 |
| `dimensionless` | a ratio or index with no unit |
| `intensity` | the raw image's own grey values |

## Network size

| Statistic | Unit | Definition |
|---|---|---|
| Total Nodes | count | Nodes of the graph the statistics read: junctions, dead ends and the nodes left along vessels, with each zero-resistance bridge contracted to one node. |
| Total Edges | vessels | Vessels (graph edges) in the network; zero-resistance bridges are not vessels and are not counted. |
| Total Edge Length (microns) | um | The sum of every vessel's centreline length (the smoothed centreline, as the haemodynamics use it). |
| Average Edge Length (microns) | um | Total Edge Length divided by Total Edges: the mean vessel length, the vessels the image faces cut short included (the Dead Ends section gives it without them). |
| Average Degree | dimensionless | The mean number of vessel ends meeting at a node (twice the vessels over the nodes). |
| Number of Branching Points | count | Junctions: nodes where three or more vessel ends meet. |
| Zero-resistance Bridges (not counted as vessels) | count | Bridges joining a vessel to a thick vessel's centreline through its lumen, modelled with negligible resistance. Contracted before every other statistic. |
| Zero-resistance Bridge Length (microns) | um | The summed length of those bridges. |
| Statistics Mode | (text) | fast: sampled or summarised where a measure is expensive; full: exact, with per-node rows. |

## Shape

| Statistic | Unit | Definition |
|---|---|---|
| Average Tortuosity Index | dimensionless | The mean over vessels of centreline length divided by the straight distance between its two end nodes; 1 is straight. |
| Average Curvature | dimensionless | The mean over vessels of (centreline length - straight distance) / centreline length: 0 for a straight vessel, nearer 1 the more it bends. Not a curvature in 1/um. |
| Tree Asymmetry Index | dimensionless | On the network's shortest-length spanning tree, rooted at its highest-degree node: the sum over nodes of the difference between their largest and smallest child subtree (in nodes), over the tree's node count. 0 for a perfectly balanced tree. |
| Fractal Dimension (Node Positions) | dimensionless | Box-counting dimension of the nodes' positions alone: the slope of log(boxes occupied) against log(1 / box size). Blind to the vessels' paths between nodes. |
| Fractal Dimension (Centreline) | dimensionless | Box-counting dimension of points every 1 um along every vessel's centreline, the usual vascular measure. |
| Fractal Dimension (Node Positions | Centreline) R^2 | dimensionless | R^2 of the log-log fit the dimension is the slope of; low means the network does not scale over the box sizes fitted. |
| Fractal Dimension (Node Positions | Centreline) Box Sizes (microns) | um | The smallest and largest box edge the dimension was fitted over (from twice the sampling step to half the network's extent, while at least 10 boxes are occupied). |
| Mean Daughter-Daughter Angle (degrees) | degrees | The mean angle between each pair of daughter vessels at a junction with a unique parent, along their tangents over the first 10 um. |
| Daughter-Daughter Angle Sample Count | count | Daughter pairs measured. |

## Density and volume

| Statistic | Unit | Definition |
|---|---|---|
| Total Vessel Length (microns) | um | The sum of every vessel's centreline length (the same as Total Edge Length). |
| Vessel Density in Tissue (microns/micron³) | um/um^3 | Total vessel length over the tissue volume: the tissue measured from the raw image when it was, else the box the network's nodes span (Tissue Volume Source says which). |
| Vessel-Occupied Volume (micron³) | um^3 | The volume of the axis-aligned box the network's nodes span. |
| Tissue Volume (micron³) | um^3 | The tissue's volume, inside the closed surface measured from the raw image (Additional measurements: tissue volume). |
| Tissue Volume Source | (text) | What the in-tissue densities divide by: the measured tissue, or the bounding box of the network's nodes. |
| Vessel Volume in Tissue (micron³) | um^3 | Segmented vessel voxels inside the measured tissue. |
| Vascular Volume Fraction in Tissue | fraction | Vessel Volume in Tissue over Tissue Volume. |
| Vessel Density in Whole Image (microns/micron³) | um/um^3 | Total vessel length over the whole image's volume, empty space included. |
| Total Image Volume (micron³) | um^3 | The image's volume: its shape times the voxel size. |
| Independent Loops Per mm^3 | 1/mm^3 | Independent loops (vessels - nodes + connected pieces) per cubic millimetre of the image. |

## Tissue volume (the tissue_volume CSV)

| Statistic | Unit | Definition |
|---|---|---|
| Tissue Volume, Voxel Count (micron³) | um^3 | The tissue's volume counted in working-grid voxels, a cross-check on the surface's volume. |
| Tissue Fraction of Image | fraction | Tissue Volume over the image's volume. |
| Tissue Surface Area (micron²) | um^2 | The area of the tissue's own surface, leaving out where the image faces cut it. |
| Tissue Surface Cut by Image Faces (micron²) | um^2 | The area where the image faces cut the tissue. |
| Tissue Pieces | count | Separate pieces of tissue kept. |
| Tissue Threshold Method | (text) | How the tissue was split from empty space: Otsu, Li, triangle or a manual intensity. |
| Tissue Threshold Split | intensity | The intensity the threshold method split the two classes at. |
| Tissue Surface Intensity | intensity | The intensity the surface passes through: halfway between the two classes' median levels. |
| Empty-Space Intensity | intensity | Median smoothed intensity of the empty-space class. |
| Tissue Intensity | intensity | Median smoothed intensity of the tissue class. |
| Tissue/Empty-Space Separation | dimensionless | Ashman's D between the two classes; low values mean the split is unreliable. |
| Tissue Working Voxel Size (z, y, x microns) | um | The voxel size of the grid the tissue was measured on, z x y x x. |

## Vessel classes (one section per class)

| Statistic | Unit | Definition |
|---|---|---|
| Vessel Classes From | (text) | How the vessels were classed: by branch-order labels (large and small arterioles, capillaries, small and large venules) or by global masks (large and small vessels). |
| Vessel Class Density Volume (micron³) | um^3 | What every class's densities divide by: the volume Vessel Density in Tissue divides by. |
| Vessel Class Density Volume Source | (text) | Where that volume came from. |
| Vessel Class Haemodynamics | (text) | Why the flow rows are missing, when they are. |
| Branch Orders | (text) | The branch-order labels of the class's vessels. |
| Vessel Count | vessels | Vessels in the class (or group). |
| Total Length (microns) | um | The class's summed centreline length. |
| Share of Network Vessel Length (fraction) | fraction | The class's length over the whole network's. |
| Mean Length (microns) | um | Mean vessel length in the class. |
| Median Length (microns) | um | Median vessel length in the class. |
| Length Density in Tissue (microns/micron³) | um/um^3 | The class's length over Vessel Class Density Volume; the classes add up to the network's density. |
| Length Density in Whole Image (microns/micron³) | um/um^3 | The class's length over the whole image's volume. |
| Mean Diameter (microns) | um | Mean of the class's vessel diameters (diameter_um, as the haemodynamics use them). |
| Median Diameter (microns) | um | Median of the class's vessel diameters. |
| Vessel Volume (micron³) | um^3 | The class's volume, a cylinder per vessel: pi (d/2)^2 L summed. |
| Volume Density in Tissue (micron³/micron³) | fraction | Vessel Volume over Vessel Class Density Volume. |
| Vessel Surface Area (micron²) | um^2 | The class's wall area, a cylinder per vessel: pi d L summed. |
| Surface Density in Tissue (micron²/micron³) | um^2/um^3 | Vessel Surface Area over Vessel Class Density Volume. |
| Mean Tortuosity Index | dimensionless | Mean centreline length over straight end-to-end distance of the class's vessels. |
| Median Tortuosity Index | dimensionless | Median centreline length over straight end-to-end distance of the class's vessels. |
| 90th Percentile Tortuosity Index | dimensionless | The 90th percentile of the same: how far the class's most winding tenth of vessels wander. |
| Vessels Carrying Flow | vessels | Vessels with a solved flow (in the flux section, above the carrying floor too). Vessels behind dead ends carry exactly none and are left out of every flow row of the class. |
| Mean Flow (nL/min) | nL/min | Mean |flow| of the class's vessels carrying flow. |
| Median Flow (nL/min) | nL/min | Median |flow| of the class's vessels carrying flow. |
| Mean Velocity (mm/s) | mm/s | Mean of each flowing vessel's flow over its lumen's cross-section (plasma mean velocity, not red-cell speed). |
| Median Velocity (mm/s) | mm/s | Median of each flowing vessel's mean velocity. |
| Mean Wall Shear Stress (Pa) | Pa | Mean of pressure drop x diameter / (4 x length), a uniform tube's wall shear stress, over the flowing vessels. |
| Median Wall Shear Stress (Pa) | Pa | Median of the same wall shear stress. |
| Mean Pressure (Pa) | Pa | Mean over the flowing vessels of each vessel's mean of its two end pressures. |

## Bifurcations and Murray's law

| Statistic | Unit | Definition |
|---|---|---|
| Murray Law Exponent | dimensionless | The exponent n the Murray ratios are taken at (3, Murray's law). |
| Mean Murray Ratio | dimensionless | The mean over junctions of sum(daughter diameter^n) / parent diameter^n; 1 is Murray's law. Only junctions with a measured diameter on every branch count. |
| Median Murray Ratio | dimensionless | The median of the same ratio. |
| Murray Ratio Sample Count | count | Junctions the ratio was taken at. |
| Murray Junctions Without Measured Diameters | count | Junctions left out because a branch had no measured diameter. |
| Median Fitted Murray Exponent | dimensionless | The median over junctions of the exponent n with parent^n = sum(daughter^n), at junctions of two or more measured daughters all narrower than the parent. |
| Fitted Murray Exponent IQR Low | dimensionless | The 25th percentile of the fitted exponents. |
| Fitted Murray Exponent IQR High | dimensionless | The 75th percentile of the fitted exponents. |
| Fitted Murray Exponent Sample Count | count | Junctions an exponent was fitted at. |

## Intercapillary distance

| Statistic | Unit | Definition |
|---|---|---|
| Mean Intercapillary Distance (microns) | um | Each vessel is sampled every 2 um; each sample's distance to the nearest vessel sharing no node with it; the vessel's value is the median along it. The mean over vessels. |
| Median Intercapillary Distance (microns) | um | The median over vessels of the same spacing. |
| Intercapillary Distance Sample Count | vessels | Vessels with a spacing. |
| Mean Intercapillary Distance Between Flowing Vessels (microns) | um | The same spacing over the vessels carrying flow alone, each to its nearest unrelated flowing vessel; the mean. |
| Median Intercapillary Distance Between Flowing Vessels (microns) | um | The median of that spacing. |
| Intercapillary Distance Between Flowing Vessels Sample Count | vessels | Flowing vessels with a spacing. |
| Flowing Over All Median Intercapillary Distance | dimensionless | The flowing vessels' median spacing over every vessel's: above 1 when the vessels not carrying flow sat between them. |

## Junction shape

| Statistic | Unit | Definition |
|---|---|---|
| Three-Vessel Junctions Measured | count | Junctions where exactly three vessel ends meet, each vessel's direction read 8 um along its centreline from the node. |
| Junctions of Four or More Vessels | count | Junctions of four or more vessel ends, not measured. |
| Junction Smallest Angle Median (degrees) | degrees | The median over three-vessel junctions of the smallest of the three angles between its vessels: 120 for a symmetric Y, near 0 where two vessels leave together. |
| Junction Smallest Angle 25th Percentile (degrees) | degrees | The 25th percentile of the same smallest angle. |
| Junction Largest Angle Median (degrees) | degrees | The median of each junction's largest angle: near 180 where one vessel runs straight on through the junction. |
| Acute Junction Share (fraction) | fraction | The share of three-vessel junctions with two vessels within 45 degrees of each other. |
| Junction Angle Sum Median (degrees) | degrees | The median of each junction's three angles summed: 360 when its vessels lie in one plane, less the further they leave it. |

## Tissue coverage

| Statistic | Unit | Definition |
|---|---|---|
| Box Within 20 um of a Vessel (fraction) | fraction | The share of the image box within 20 um of a vessel centreline, read on a 10 um grid of points (200,000 points at random, seeded, in a larger box). |
| Box Beyond 50 um of a Vessel (fraction) | fraction | The share of the same points further than 50 um from every vessel centreline: the holes in the network. |
| Median Distance to Nearest Vessel (microns) | um | The median over the same points of the distance to the nearest vessel centreline. |
| 90th Percentile Distance to Nearest Vessel (microns) | um | The 90th percentile of that distance: how far the network's larger holes reach. |
| Coverage Box | (text) | The box the points fill: the image the run read, or the box the vessels span when the graph does not say. |
| Coverage Probe Points | count | Points the coverage was read at. |

## Vessel columns

| Statistic | Unit | Definition |
|---|---|---|
| Column Vessels (within 30 degrees of z) | vessels | Vessels whose straight end-to-end line is at least 10 um long and within 30 degrees of the stack (z) axis. |
| Column Share of Vessel Length (fraction) | fraction | The columns' centreline length over every vessel's. |
| Columns per mm² of Image Plane | 1/mm^2 | Columns per square millimetre of the image's y-x plane. |
| Column Spacing Median (microns) | um | The median distance in the y-x plane from each column's midpoint to the nearest other column's. |
| Column Spacing Clark-Evans Ratio | dimensionless | The columns' mean nearest-neighbour distance in the y-x plane over a random pattern's at the same density: 1 at random, above 1 evenly spaced, below 1 bunched. No edge correction. |
| Median Column Tortuosity Index | dimensionless | The median of the columns' centreline length over their straight end-to-end distance. |

## Dead ends

| Statistic | Unit | Definition |
|---|---|---|
| Dead Ends | count | Dead ends: nodes only one vessel reaches. |
| Dead Ends Leaving the Image | count | Dead ends whose tip touches an image face, or whose segmented vessel runs on to one: open in the tissue, never in a dead-end density. |
| Dead Ends at a Thick Vessel's Wall (not joined) | count | Dead ends stopping within the thick-vessel split's wall-join reach (15 um) of a thick vessel's lumen, off its centreline, that the split did not join to it. |
| Dead Ends the Graph Stopped Short of (mask runs on) | count | Dead ends where the segmented vessel runs on past the tip. |
| Dead Ends of Vessels Broken by the Segmentation | count | Dead ends with another vessel straight ahead across one stretch of background. |
| Dead Ends That Are Artefacts | count | Dead ends off the mask, inside a neighbouring lumen or along a vessel's wall. |
| Blind Ends (marked by hand) | count | Dead ends a person marked as real blind ends. |
| Interior Dead Ends, Blind or Broken (unknown) | count | Interior dead ends no rule could class. |
| Ends of Pieces Without a Junction | count | Dead ends of pieces of network with no junction. |
| Interior Dead Ends per mm of Vessel (1/mm) | 1/mm | Interior dead ends that could be the tissue's own (blind, unknown, broken, stopped short, at a thick vessel's wall) per mm of vessel. |
| Interior Dead Ends per mm^3 of Tissue (1/mm^3) | 1/mm^3 | The same dead ends per cubic millimetre of measured tissue. |
| Interior Dead-End Chain Length, median (microns) | um | The median length of those dead ends' chains, tip to junction. |
| Interior Dead-End Chain Length, 10th / 90th percentile (microns) | um | The 10th and 90th percentiles of those chain lengths. |
| Vessel Length on Chains Cut by the Image Faces (microns) | um | Length of the chains ending at a dead end that leaves the image: lengths the image censored. |
| Share of Vessel Length Cut by the Image Faces (fraction) | fraction | That length over the network's. |
| Vessels Cut by the Image Faces | vessels | Vessels on those chains: the image cut them short, so each length is a lower bound (decision D5). The VTK's cut_by_image_face and the connectivity CSV flag each one. |
| Mean Vessel Length, Not Cut by the Image Faces (microns) | um | The mean vessel length over the vessels the image faces did not cut: the uncensored lengths. |
| Median Vessel Length, Not Cut by the Image Faces (microns) | um | Their median length. |
| Mean Vessel Length, Cut by the Image Faces, Censored (microns) | um | The mean length of the vessels the image faces cut short, as far as the image shows them. |
| Median Vessel Length, Cut by the Image Faces, Censored (microns) | um | Their median length. |
| Pieces Without a Junction | count | Pieces of network with no junction (lone vessels and chains). |
| Length of Pieces Without a Junction (microns) | um | Their summed length. |
| Vessel Length Carrying Flow (microns) | um | Length the solve gave a flow. |
| Vessel Length Behind Interior Dead Ends, No Flow (microns) | um | Length that carries no flow because it leads only to dead ends inside the tissue. |
| Vessel Length Behind Closed Image-Face Ends, No Flow (microns) | um | Length that carries no flow because it leads only to image-face ends with no boundary condition. |
| Vessel Length Without Inlet and Outlet, No Flow (microns) | um | Length in pieces with no inlet and outlet, where no flow is defined. |

## Robustness

| Statistic | Unit | Definition |
|---|---|---|
| Bridge Edge Count | vessels | Vessels whose occlusion alone splits the network into more pieces (graph bridges). |
| Bridge Edge Fraction | fraction | Bridge vessels over all vessels. |
| Articulation Point Count | count | Nodes whose removal splits the network into more pieces. |
| Articulation Point Fraction | fraction | Articulation points over all nodes. |
| Critical Bridge Edge Count | vessels | Bridges whose occlusion separates some inlet from some outlet. |
| Critical Bridge Edge Fraction | fraction | Critical bridges over all bridges. |
| Critical Articulation Point Count | count | Articulation points whose removal separates some inlet from some outlet. |
| Critical Articulation Point Fraction | fraction | Critical articulation points over all articulation points. |

## Inlet-outlet routes

| Statistic | Unit | Definition |
|---|---|---|
| Bottleneck Min-Cut Edge Count | vessels | Vessels in the minimum cut separating every inlet from every outlet. |
| Bottleneck Min-Cut Capacity | given by the row Bottleneck Min-Cut Capacity Unit | The capacity of that cut: its vessels under topology or length weighting, their summed conductance under resistance weighting. |
| Bottleneck Min-Cut Capacity Unit | (text) | The unit of the capacity. |
| Bottleneck Route Share Max | fraction | The largest share of shortest inlet-outlet routes (one per connected pair) passing through one vessel; 1 on a trunk every route uses. |
| Bottleneck Route Share Mean | fraction | The mean of that share over vessels. |
| Bottleneck Top Edges | (text) | The vessels with the largest route share: [node, node, key, share]. |
| Bottleneck Route Share Method | (text) | exact, or the inlets sampled as route sources. |
| Shunt Route Fraction Threshold | fraction | A vessel is a shunt when the shortest inlet-outlet route through it is at most this share of the median one. |
| Shunt Edge Count | vessels | Vessels on a shunt route. |
| Shunt Edge Fraction | fraction | Shunt vessels over all vessels. |
| Shunt Pathway Count | count | Connected groups of shunt vessels joining an inlet to an outlet. |
| Shortest Inlet-Outlet Route | by the route weighting (Shunt Route Weighting): resistance Pa.s/m^3, length um, topology vessels | The shortest inlet-outlet route through any vessel, measured by the route weighting. |
| Median Inlet-Outlet Route | by the route weighting (Shunt Route Weighting): resistance Pa.s/m^3, length um, topology vessels | The median over vessels of the shortest inlet-outlet route through each: the reference a shunt is judged against. |
| Shunt Outflow Fraction | fraction | The share of the solved flow at the outlets that leaves through shunt vessels. |

## Current flow

| Statistic | Unit | Definition |
|---|---|---|
| Equivalent Inlet-Outlet Resistance | given by the row Equivalent Inlet-Outlet Resistance Unit | The resistance between all inlets and all outlets with the route weighting's conductances (1 / flow at a unit pressure difference). |
| Equivalent Inlet-Outlet Resistance Unit | (text) | The unit of that resistance. |
| Current Flow Share Max | fraction | The largest share of the total inlet-outlet flow passing through one vessel when flow splits over every route by its conductance (Kirchhoff). |
| Current Flow Share Mean | fraction | The mean of that share over vessels. |
| Current Flow Top Vessels | (text) | The vessels with the largest share: [node, node, key, share]. |
| Kirchhoff Index | given by the row Equivalent Inlet-Outlet Resistance Unit | The sum of resistance distances over every node pair of the largest connected piece (full mode, up to 3,000 nodes). |

## Occlusion

| Statistic | Unit | Definition |
|---|---|---|
| Occlusion Impact Method | (text) | exact, or how many vessels were evaluated. |
| Occlusion Impact Assumptions | (text) | What the occlusion model leaves out. |
| Occlusion Curves Assumptions | (text) | What the occlusion model leaves out. |
| Occlusion Hypoperfusion Threshold (fraction) | fraction | A vessel counts as hypoperfused when its flow falls by at least this share. |
| Occlusion Hypoperfused Length Unit | (text) | The unit of the hypoperfused lengths: um, or vessels when a vessel has no length. |
| Occlusion Impact (Network) / By Vessel Type / By Branch Order | (text) | Sections: the single-vessel occlusion summary for the whole network, per vessel type and per branch order. |
| Evaluated Count | vessels | Vessels whose single occlusion was evaluated. |
| Mean Flow Loss | fraction | Blocking one vessel alone: the share of total inlet-outlet flow lost (with the route weighting's fixed conductances). The mean over vessels. |
| Median Flow Loss | fraction | The median of that flow loss. |
| P90 Flow Loss | fraction | The 90th percentile of that flow loss. |
| Max Flow Loss | fraction | The largest flow loss from blocking one vessel. |
| Mean Hypoperfused Length | given by the row Occlusion Hypoperfused Length Unit | Blocking one vessel alone: the length of other vessels whose flow falls by at least the hypoperfusion threshold. The mean over vessels. |
| Max Hypoperfused Length | given by the row Occlusion Hypoperfused Length Unit | The largest such length from blocking one vessel. |
| Most Critical Vessel | (text) | The vessel whose occlusion loses the most flow: [node, node, key, flow loss]. |
| Top Vessels | (text) | The vessels whose occlusion loses the most flow: [node, node, key, flow loss]. |
| Vessels Losing Over N% Of Flow | vessels | Vessels whose occlusion alone loses more than that share of total inlet-outlet flow. |
| Occlusion Curve (random | highest_current_flow_share | narrowest_diameter) | (text) | Sections: vessels blocked cumulatively, at random (seeded, averaged), busiest first, or narrowest first; or why a curve is missing. |
| Robustness Index | fraction | The mean remaining flow, as a share of the intact network's, across the removal range; 1 is unaffected. |
| Removal Fraction At 50% Flow | fraction | The share of vessels removed at which flow halves. |
| Curve [fraction removed, flow fraction, outlets reachable] | (text) | The curve itself: share of vessels removed, remaining flow share, share of outlets still connected to an inlet. |

## Perfusion territories

| Statistic | Unit | Definition |
|---|---|---|
| Arteriolar / Venular Territory Count | count | Inlets (arteriolar) or outlets (venular) that reach at least one vessel first: territories. |
| Arteriolar / Venular Watershed Vessel Count | vessels | Vessels on a border between two territories: their two ends are reached first from different inlets (or outlets). |
| Arteriolar / Venular Watershed Length Fraction | fraction | Watershed vessels' length over the network's. |
| Arteriolar / Venular Unreached Vessel Count | vessels | Vessels no inlet (or outlet) reaches. |
| Arteriolar / Venular Territory Size Min / Median / Max (vessels) | vessels | The smallest, median and largest territory, in vessels. |
| Arteriolar / Venular Territory Length Median | um | The median territory's summed vessel length. |

## Capillary bed joins

| Statistic | Unit | Definition |
|---|---|---|
| Capillary Beds | count | Capillary beds, one per feed (the nodes capillary branch orders are counted from). |
| Capillary Beds Joined | count | Beds that meet another bed on a watershed vessel. |
| Capillary Beds Joined (fraction) | fraction | Beds joined over all beds. |
| Capillary Bed Join Order, Kaplan-Meier Median | dimensionless | A bed's join order is the lowest capillary branch order on a watershed vessel with another bed; the Kaplan-Meier median over beds, with beds that never join censored. |
| Capillary Bed Join Distance, Kaplan-Meier Median (microns) | um | The same Kaplan-Meier median of the route length from the feed to where its bed joins another. |
| Capillary Beds Censored, Leaving the Image / Draining to a Venule / Ending Blind | count | Beds that never join another, by why: they leave the image, drain into a venule, or end blind. |
| Capillary Beds Joining Within Their Arteriolar Tree | count | Joins between beds fed from the same inlet's tree. |
| Capillary Beds Joining Another Arteriolar Tree | count | Joins between beds fed from different inlets' trees. |
| Capillaries No Feed Reaches | vessels | Capillaries no feed reaches. |
| Graph Built With Short Loops Removed Under (microns) | um | remove_short_loops_under_um when the graph was built: loops shorter than this were broken, so the earliest joins may be missing. |

## Transit time

| Statistic | Unit | Definition |
|---|---|---|
| Transit Time Assumptions | (text) | What the transit-time model assumes. |
| Mean Transit Time (s) | s | The mean time blood takes from the inlets to the outlets, each vessel's time its volume over its flow, mixed by flow at every junction. Open ends feeding or draining the network count as inlets and outlets; vessels behind dead ends carry no flow and take no part. |
| Transit Time SD (s) | s | The standard deviation of that inlet-to-outlet time. |
| Transit Time CV | dimensionless | Transit Time SD over Mean Transit Time. |
| Capillary Mean Transit Time (s) | s | The mean of the same time spent in capillaries alone. |
| Capillary Transit Time Heterogeneity CTH (s) | s | The standard deviation of the capillary transit time (CTH). |
| Capillary Transit Time CV | dimensionless | CTH over the capillary mean transit time. |
| Vessel Transit Time By Vessel Type | (text) | Section: each vessel type's single-vessel transit times. |
| Flow-Weighted Mean (s) | s | The flow-weighted mean of the type's single-vessel transit times (volume over flow). |
| Median (s) | s | The median of the type's single-vessel transit times. |

## Flux heterogeneity and red cells

| Statistic | Unit | Definition |
|---|---|---|
| Flux Heterogeneity Assumptions | (text) | What the rows assume. |
| Carrying Flow Floor (m³/s) | m^3/s | A vessel carries flow when the solve reached it and its |flow| is above this: 1e-9 of the network's largest. Vessels behind dead ends carry exactly none, so no row of this section counts them. |
| Red Cell Volume (fL) | fL | The mean red-cell volume the red-cell rows count cells by (red_cell_volume_fl). |
| Haematocrit Source | (text) | Which discharge haematocrit the red-cell rows used: the vessels' own, or the run's one. |
| Vessels | vessels | Vessels in the group. |
| Vessels / Capillaries Carrying Flow | vessels | Vessels (or capillaries) carrying flow. |
| Vessels / Capillaries Carrying Flow Fraction | fraction | Those over all the group's vessels. |
| Vessels / Capillaries Carrying Flow Without Diameter | vessels | Carrying vessels with no diameter, left out of the velocity rows. |
| Vessels / Capillaries Carrying Flow Without Haematocrit | vessels | Carrying vessels with no haematocrit, left out of the red-cell rows. |
| Flowing Vessel / Capillary Length (microns) | um | The carrying vessels' summed length. |
| Flowing Vessel / Capillary Length Fraction | fraction | That length over the group's. |
| Flowing Vessel / Capillary Volume (micron³) | um^3 | The carrying vessels' volume, a cylinder per vessel. |
| Flowing Vessel / Capillary Volume Fraction | fraction | That volume over the group's. |
| Vessel / Capillary Flow Mean / SD (nL/min) | nL/min | Mean and sample standard deviation of |flow| over the carrying vessels. |
| Vessel / Capillary Velocity Mean / SD (mm/s) | mm/s | Mean and sample standard deviation of mean velocity (flow over the lumen's cross-section) over the carrying vessels. |
| Vessel / Capillary Red Cell Flux Mean / SD (cells/s) | cells/s | Mean and sample standard deviation of the predicted red-cell flux, flow x discharge haematocrit / red-cell volume. A prediction, not a count. |
| Vessel / Capillary Red Cell Velocity Mean / SD (mm/s) | mm/s | Mean and sample standard deviation of the predicted red-cell velocity, mean velocity x discharge / tube haematocrit (Fahraeus, Pries et al. 1990). |
| Vessel / Capillary Tube Haematocrit Mean | fraction | The mean predicted tube haematocrit: the haematocrit inside the vessel, below the discharge haematocrit (Fahraeus, Pries et al. 1990). |
| Vessel / Capillary Red Cell Linear Density Mean (cells/mm) | cells/mm | The mean predicted spacing of red cells: tube haematocrit x cross-section / red-cell volume, cells per mm of vessel. |
| Vessel / Capillary Flow CV | dimensionless | The coefficient of variation (sample SD over mean) of |flow| over the carrying vessels. |
| Vessel / Capillary Velocity CV | dimensionless | The coefficient of variation of mean velocity, flow over the lumen's cross-section (plasma, not red-cell speed), over the carrying vessels. |
| Vessel / Capillary Red Cell Flux CV | dimensionless | The coefficient of variation of the predicted red-cell flux over the carrying vessels; with one haematocrit everywhere it is the flow's CV. |
| Vessel / Capillary Red Cell Velocity CV | dimensionless | The coefficient of variation of the predicted red-cell velocity (Fahraeus, Pries et al. 1990) over the carrying vessels. |
| Capillary Flux Heterogeneity Status | (text) | Why the capillary rows are missing. |
| Flux Heterogeneity By Vessel Type | (text) | Section: the same rows per vessel type. |

## Loops

| Statistic | Unit | Definition |
|---|---|---|
| Cyclomatic Number | count | Independent loops: vessels - nodes + connected pieces. |
| Cyclomatic Number Per Node | dimensionless | The cyclomatic number over the node count. |
| Loop Length Weighting | (text) | What loop lengths are measured in: length (um), or vessels where a vessel has no length. |
| Loop Method | (text) | exact, or how many junction pairs were sampled. |
| Vessels On No Loop | vessels | Vessels no loop runs through (bridges). |
| Vessels On No Loop Fraction | fraction | Those over the vessels evaluated. |
| Loop Length P10 / P25 / Median / P75 / P90 (um or vessels) | as named (um or vessels) | Percentiles over vessels of the shortest loop through each: the vessel plus the shortest detour between its ends. |
| Median Loop Length By Vessel Type | (text) | Section: the median shortest loop through each vessel type's vessels. |

## Strahler orders

| Statistic | Unit | Definition |
|---|---|---|
| Strahler Method | (text) | The arterial and venous trees, or the whole network's tree from the inlets (approximate) without an arteriole/venule classification. |
| Strahler Status | (text) | Why no orders were found. |
| Strahler Arterial / Venous / Network | (text) | Sections: one tree's Strahler orders, or why it has none. |
| Max Order | count | The tree's highest Strahler order. |
| Orders | (text) | Section: one row per Strahler order. |
| Element Count | count | Elements of that order: runs of consecutive vessels of one order. |
| Mean Element Length (um or vessels) | as named (um or vessels) | The mean length of the order's elements. |
| Mean Element Diameter (um) | um | The mean length-weighted diameter of the order's elements. |
| Bifurcation Ratio R_B | dimensionless | Horton's ratio of element counts between successive orders, from a log-linear fit. |
| Length Ratio R_L | dimensionless | Horton's ratio of mean element lengths between successive orders. |
| Diameter Ratio R_D | dimensionless | Horton's ratio of mean element diameters between successive orders. |
| Unordered Vessels (anastomoses or unreached) | vessels | The tree's vessels with no order: closing a loop within it, or not reached from its roots. |

## Algebraic connectivity

| Statistic | Unit | Definition |
|---|---|---|
| Algebraic Connectivity lambda_2 | by the route weighting (Algebraic Connectivity Route Weighting): resistance m^3/(Pa.s), length 1/um, topology dimensionless | The second-smallest eigenvalue of the conductance-weighted Laplacian of the largest connected piece: 0 when it falls apart, larger the harder it is to cut in two. |
| Normalised Algebraic Connectivity | dimensionless | lambda_2 over the mean weighted degree, comparable across networks. |
| Algebraic Connectivity Component Nodes | count | Nodes in the piece lambda_2 was found on. |
| Fiedler Split Vessel Counts [A, B] | vessels | Vessels on each side of the Fiedler vector's split, larger side first. |
| Fiedler Cut Vessel Count | vessels | Vessels the Fiedler split cuts. |

## Topology

| Statistic | Unit | Definition |
|---|---|---|
| Degree Assortativity | dimensionless | The correlation between the degrees of the two nodes of each vessel, -1 to 1; vascular trees are usually below 0. |
| Rich Club Coefficient | dimensionless | The density of vessels among nodes of degree above the threshold (unnormalised), at the highest threshold any node passes. |
| Rich Club Coefficient Degree Threshold | count | That degree threshold. |
| Max Core Number | count | The highest k for which a k-core (every node with at least k neighbours inside it) exists. |
| Mean Core Number | count | The mean over nodes of each node's core number. |
| Innermost Core Size | count | Nodes in the innermost core. |
| Flow Hierarchy | fraction | The share of vessels, oriented by their solved flow, on no directed cycle; 1 means the flow never recirculates. |
| Flow Hierarchy Edges Skipped | vessels | Vessels with no determinate flow direction, left out: those the solve's flow state says carry no flow (behind dead ends, in pieces with no inlet and outlet) and those with zero or no flow. |
| Flow Hierarchy Vessels Behind Dead Ends (excluded) | vessels | Of those, the vessels behind dead ends (inside the tissue, or at an image face with no boundary condition): their flow is exactly 0, so they have no direction. |
| Path Efficiency | 1/um | 1 over the mean shortest-path length between node pairs. |
| Average Shortest Path Length (microns) | um | The mean shortest path, by centreline length, between node pairs. |
| Path Efficiency Pair Sample Size | count | Node pairs measured. |
| Path Efficiency Pair Coverage | fraction | Node pairs measured over all node pairs. |
| Community Count | count | Communities found by greedy modularity on the network's topology. |
| Largest Community Size | count | Nodes in the largest community. |
| Mean Community Size | count | The mean nodes per community. |
| Community Method | (text) | How the communities were found. |
| communities | (text) | Full mode: every community's nodes. |
| Betweenness Mean | fraction | The mean over nodes of their betweenness centrality: the share of shortest paths (counted in vessels) between other node pairs passing through the node. |
| Betweenness Max | fraction | The largest node betweenness centrality. |
| Betweenness Top Nodes | (text) | The nodes of highest betweenness: [node, value]. |
| Betweenness Method | (text) | exact, or the number of sampled sources. |
| A node ID | fraction | Full mode: that node's betweenness centrality. |

## Haemodynamics

| Statistic | Unit | Definition |
|---|---|---|
| Two-Point Equivalent Resistance (Pa.s/m^3) | Pa.s/m^3 | The resistance between the first inlet and the first outlet: the pressure difference driving a unit flow between them alone. |
| Two-Point Resistance Nodes | (text) | The two nodes it was found between. |
| Network Resistance, Pressure Drop / Total Inflow (Pa.s/m^3) | Pa.s/m^3 | Inlet pressure minus outlet pressure over the total flow in through every designated inlet; the flow through open ends is reported beside it, not in it. |
| Open-End Model | (text) | The boundary condition on vessels leaving the image (open_end_model): estimated, terminal_resistance or outlet. Written only when open ends were given one. |
| Open Ends Given a Condition | count | Dead ends leaving the image, in a piece with a designated inlet or outlet, that the solve gave the boundary condition. |
| Inflow Through Open Ends (m^3/s) | m^3/s | The flow entering the network through those open ends. |
| Outflow Through Open Ends (m^3/s) | m^3/s | The flow leaving the network through them. |

## Capillary bed table (the capillary_bed_joins CSV's columns)

| Statistic | Unit | Definition |
|---|---|---|
| bed | (text) | The bed's label, its feed's (F01, F02, ...). |
| feed_node | (text) | The node the bed is fed from. |
| arteriolar_tree | (text) | The arteriolar tree (inlet) the feed belongs to. |
| vessel_count | vessels | Capillaries in the bed. |
| bed_length_um | um | The bed's summed capillary length. |
| join_order | dimensionless | The lowest capillary branch order on a watershed vessel with another bed; empty for a censored bed. |
| join_distance_um | um | The route length from the feed to where the bed first joins another; empty for a censored bed. |
| censored | (text) | True when the bed never joins another. |
| censor_reason | (text) | Why: it leaves the image, drains into a venule, or ends blind. |
| joins_bed | (text) | The bed it first joins. |
| join_class | (text) | within_tree when the two beds are fed from one inlet's tree, between_trees otherwise. |

## Rows every analysis may write

| Statistic | Unit | Definition |
|---|---|---|
| (analysis) Route Weighting | (text) | The conductance model a route analysis used -- topology (one per vessel), length or resistance -- and any fallback; it sets the unit of the analysis's lengths and resistances. |
| (analysis) Status | (text) | Why an analysis or section reported nothing (N/A and its reason). |
| (analysis) Nodes Both Inlet And Outlet (excluded) | count | Nodes given as both inlet and outlet, left out of the analysis. |
| Flow Rows Leave Out the Image-Face Margin (microns) | um | With statistics_flow_face_margin_um: the vessels this close to an image face are left out of the section's flow rows, as though they carried no flow. |
| Vessels Left Out Within the Image-Face Margin | vessels | How many vessels that margin left out of the section's flow rows. |
| (analysis) Open Ends Taken As Inlets | count | Open ends leaving the image, given a boundary condition by the solve (open_end_model), through which blood enters the network: the analysis took them as inlets beside the designated ones. |
| (analysis) Open Ends Taken As Outlets | count | Open ends through which blood leaves the network, taken as outlets beside the designated ones. |
| arteriole, capillary, venule ... (a vessel type) | by the route weighting (Loop Length Weighting): resistance vessels, length um, topology vessels | Under Median Loop Length By Vessel Type: the median shortest loop through that type's vessels. |

## Branch order statistics (the branch_statistics CSV's columns)

| Statistic | Unit | Definition |
|---|---|---|
| Branch Order | (text) | The branch-order label (Large_Art, Art, BO for capillaries, Ven, Large_Ven, and a number). |
| Edge Count | vessels | Vessels of that branch order. |
| Tortuosity Sample Count | vessels | Vessels whose tortuosity was measured. |
| Mean Emergence Angle (degrees) | degrees | The mean angle between the parent's and each daughter's tangent where a vessel of that order leaves its parent. |
| Emergence Angle Sample Count | count | Emergence angles measured. |
| Mean Pressure Drop (Pa) | Pa | The mean pressure drop along the order's vessels carrying flow; vessels behind dead ends and in pieces with no inlet and outlet are left out. |
| Total Pressure Drop (Pa) | Pa | The summed pressure drop along the same vessels. |
| Pressure Drop Sample Count | vessels | Vessels with a pressure drop, those behind dead ends left out. |
| Diameter Coefficient of Variation | dimensionless | The standard deviation over the mean of the order's diameters. |
| Diameter Sample Count | vessels | Vessels with a diameter. |
