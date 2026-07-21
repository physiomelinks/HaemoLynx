import sys

with open("examples/carotid_image_to_model.py", "r") as f:
    content = f.read()

# Replace monolithic preprocessing with map-reduce Phase 2
old_block = """        image, binary = _preprocess_local_mask(
            raw_prob_map, entropy_map, pre_config, skel_config, graph_config, pipeline_config,
            optimize_trials=args.optimize_preprocessing, optimize_patience=pipeline_config.optimize_patience
        )"""

new_block = """        if getattr(pipeline_config, 'chunk_fraction', None) is not None and pipeline_config.chunk_fraction < 1.0:
            print(f"\\n--- Launching Map-Reduce Preprocessing Architecture (fraction={pipeline_config.chunk_fraction}) ---")
            from ImageLynx.pipeline.map_reduce import map_reduce_pipeline
            
            def preprocess_local_chunk(chunk_raw_prob, bbox):
                import copy
                local_pre_config = copy.deepcopy(pre_config)
                
                core_z1, core_z2, core_y1, core_y2, core_x1, core_x2 = bbox['padded']
                local_entropy = entropy_map[core_z1:core_z2, core_y1:core_y2, core_x1:core_x2] if entropy_map is not None else None
                
                _, local_binary = _preprocess_local_mask(
                    chunk_raw_prob, local_entropy, local_pre_config, skel_config, graph_config, pipeline_config,
                    optimize_trials=args.optimize_preprocessing, optimize_patience=pipeline_config.optimize_patience
                )
                
                # Strip overlap margin
                pz1, pz2, py1, py2, px1, px2 = bbox['padded']
                cz1, cz2, cy1, cy2, cx1, cx2 = bbox['core']
                
                # Calculate relative core slices within the padded array
                rel_z1 = cz1 - pz1
                rel_z2 = rel_z1 + (cz2 - cz1)
                rel_y1 = cy1 - py1
                rel_y2 = rel_y1 + (cy2 - cy1)
                rel_x1 = cx1 - px1
                rel_x2 = rel_x1 + (cx2 - cx1)
                
                local_core_binary = local_binary[rel_z1:rel_z2, rel_y1:rel_y2, rel_x1:rel_x2]
                return local_core_binary

            binary = map_reduce_pipeline(
                volume=raw_prob_map,
                chunk_fraction=pipeline_config.chunk_fraction,
                margin=pipeline_config.margin,
                worker_fn=preprocess_local_chunk,
                n_jobs=pipeline_config.n_jobs
            )
            image = raw_prob_map # Monolithic image isn't strictly needed for graph, but we pass the raw map
            
        else:
            image, binary = _preprocess_local_mask(
                raw_prob_map, entropy_map, pre_config, skel_config, graph_config, pipeline_config,
                optimize_trials=args.optimize_preprocessing, optimize_patience=pipeline_config.optimize_patience
            )

        if getattr(pipeline_config, 'exit_after_mask', False):
            import sys
            import pyvista as pv
            import ImageLynx.io as io
            print(f"\\n--- Exporting Globally Stitched Vessel Mask ---")
            spacing = io.get_tif_spacing(image_path) if input_format == "tif" else (1.0, 1.0, 1.0)
            
            vtk_vol = pv.ImageData()
            vtk_vol.dimensions = np.array(binary.shape)
            vtk_vol.spacing = (spacing[2], spacing[1], spacing[0])
            vtk_vol.point_data["vessel_mask"] = binary.flatten(order="F").astype(np.uint8)
            
            out_path = pipeline_config.vtk_output_prefix.with_name(f"{pipeline_config.vtk_output_prefix.name}_vessel_mask.vti")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            vtk_vol.save(out_path)
            
            print(f"Exported Vessel Mask to: {out_path}")
            print("Exiting pipeline early as requested (--exit-after-mask).")
            sys.exit(0)"""

content = content.replace(old_block, new_block)

with open("examples/carotid_image_to_model.py", "w") as f:
    f.write(content)
