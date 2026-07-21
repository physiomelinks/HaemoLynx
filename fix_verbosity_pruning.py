import sys

# 1. Update map_reduce_pipeline
with open("src/ImageLynx/pipeline/map_reduce.py", "r") as f:
    mr_content = f.read()

mr_old = """    def process_chunk(bbox):
        pz1, pz2, py1, py2, px1, px2 = bbox['padded']
        chunk = volume[pz1:pz2, py1:py2, px1:px2]
        
        # If the chunk is entirely empty background, skip processing
        if np.max(chunk) == 0:
            return None, bbox
            
        # Execute local pipeline (returns local_core_binary without margin)
        local_core_binary = worker_fn(chunk, bbox)
        
        return local_core_binary, bbox

    print(f"Executing {len(bboxes)} chunks across {n_jobs if n_jobs > 0 else 'all'} workers...")
    
    # Phase 2: Parallel Local Execution
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_chunk)(bbox) for bbox in bboxes
    )"""

mr_new = """    def process_chunk(item):
        idx, bbox = item
        pz1, pz2, py1, py2, px1, px2 = bbox['padded']
        chunk = volume[pz1:pz2, py1:py2, px1:px2]
        
        # If the chunk is entirely empty background, skip processing
        if np.max(chunk) == 0:
            return None, bbox
            
        # Execute local pipeline (returns local_core_binary without margin)
        local_core_binary = worker_fn(chunk, bbox, idx, len(bboxes))
        
        return local_core_binary, bbox

    print(f"Executing {len(bboxes)} chunks across {n_jobs if n_jobs > 0 else 'all'} workers...")
    
    # Phase 2: Parallel Local Execution
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_chunk)(item) for item in enumerate(bboxes, 1)
    )"""

mr_content = mr_content.replace(mr_old, mr_new)
with open("src/ImageLynx/pipeline/map_reduce.py", "w") as f:
    f.write(mr_content)


# 2. Update carotid_image_to_model.py
with open("examples/carotid_image_to_model.py", "r") as f:
    main_content = f.read()

# Update _preprocess_local_mask signature
pre_old = "def _preprocess_local_mask(raw_prob_map, entropy_map, pre_config, skel_config, graph_config, pipeline_config, optimize_trials=0, optimize_patience=15):"
pre_new = "def _preprocess_local_mask(raw_prob_map, entropy_map, pre_config, skel_config, graph_config, pipeline_config, optimize_trials=0, optimize_patience=15, chunk_idx=1, total_chunks=1):"
main_content = main_content.replace(pre_old, pre_new)

# Add optuna print statement
opt_old = """    if optimize_trials > 0:
        import ImageLynx.statistics.benchmarking as benchmarking
        import ImageLynx.statistics.auto_tuner as auto_tuner
        import copy
        def pre_eval_callback(suggested_kwargs):"""
opt_new = """    if optimize_trials > 0:
        import ImageLynx.statistics.benchmarking as benchmarking
        import ImageLynx.statistics.auto_tuner as auto_tuner
        import copy
        
        print(f"\\n--- [Chunk {chunk_idx}/{total_chunks}] Launching Optuna Preprocessing Auto-Tuner ({optimize_trials} trials) ---", flush=True)

        def pre_eval_callback(suggested_kwargs):"""
main_content = main_content.replace(opt_old, opt_new)

# Update worker_fn in main script
worker_old = """            def preprocess_local_chunk(chunk_raw_prob, bbox):
                import copy
                local_pre_config = copy.deepcopy(pre_config)
                
                core_z1, core_z2, core_y1, core_y2, core_x1, core_x2 = bbox['padded']
                local_entropy = entropy_map[core_z1:core_z2, core_y1:core_y2, core_x1:core_x2] if entropy_map is not None else None
                
                _, local_binary = _preprocess_local_mask(
                    chunk_raw_prob, local_entropy, local_pre_config, skel_config, graph_config, pipeline_config,
                    optimize_trials=args.optimize_preprocessing, optimize_patience=pipeline_config.optimize_patience
                )"""

worker_new = """            def preprocess_local_chunk(chunk_raw_prob, bbox, chunk_idx, total_chunks):
                import copy
                local_pre_config = copy.deepcopy(pre_config)
                
                core_z1, core_z2, core_y1, core_y2, core_x1, core_x2 = bbox['padded']
                local_entropy = entropy_map[core_z1:core_z2, core_y1:core_y2, core_x1:core_x2] if entropy_map is not None else None
                
                _, local_binary = _preprocess_local_mask(
                    chunk_raw_prob, local_entropy, local_pre_config, skel_config, graph_config, pipeline_config,
                    optimize_trials=args.optimize_preprocessing, optimize_patience=pipeline_config.optimize_patience,
                    chunk_idx=chunk_idx, total_chunks=total_chunks
                )"""
main_content = main_content.replace(worker_old, worker_new)

# Add global pruning after map_reduce_pipeline
global_prune_old = """            binary = map_reduce_pipeline(
                volume=raw_prob_map,
                chunk_fraction=pipeline_config.chunk_fraction,
                margin=pipeline_config.margin,
                worker_fn=preprocess_local_chunk,
                n_jobs=pipeline_config.n_jobs
            )
            image = raw_prob_map # Monolithic image isn't strictly needed for graph, but we pass the raw map
            
        else:"""

global_prune_new = """            binary = map_reduce_pipeline(
                volume=raw_prob_map,
                chunk_fraction=pipeline_config.chunk_fraction,
                margin=pipeline_config.margin,
                worker_fn=preprocess_local_chunk,
                n_jobs=pipeline_config.n_jobs
            )
            
            if skel_config.prune_mask_before > 0:
                print(f"Applying global pruning to stitched mask (keeping top {skel_config.prune_mask_before} components)...")
                import ImageLynx.preprocessing as preprocessing
                binary = preprocessing.skeleton.keep_largest_mask_components(
                    binary, n_components=skel_config.prune_mask_before, connectivity=skel_config.component_connectivity
                )
                
            image = raw_prob_map # Monolithic image isn't strictly needed for graph, but we pass the raw map
            
        else:"""
main_content = main_content.replace(global_prune_old, global_prune_new)

with open("examples/carotid_image_to_model.py", "w") as f:
    f.write(main_content)

