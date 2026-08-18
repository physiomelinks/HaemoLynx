// FIJI Batch Pre-processing Macro for WKY & SHR Carotid Body CFM Image Cohorts
// Optimized for preparing 3D Z-stacks for by-hand labeling in Ilastik Pixel Classification.
// 
// Steps Automated:
// 1. Axial Intensity & Bleach Correction (via Histogram Matching)
// 2. 3D Noise Reduction & Smoothing (via Background Subtraction & 3D Median Filter)
// 3. Automated Subregion Tiling (RAM Optimization for Ilastik, keeping full Z-depth intact)
// 4. Intensity Standardization & Local Contrast Normalization per Tile
//
// Requirements: 
// - Run this on a computer with Fiji/ImageJ installed.
// - Ensure "3D ImageJ Suite" and "IJBP-Plugins (MorphoLibJ)" are enabled in Help > Update > Manage update sites.

#javascript

// Define Dialog UI for User Inputs
Dialog.create("CB 3D CFM Batch Pre-processing Parameters");
Dialog.addMessage("Select your pre-processing parameters below:");
Dialog.addNumber("Rolling Ball Radius (pixels):", 30);
Dialog.addNumber("Median Filter 3D Radius (X, Y):", 1);
Dialog.addNumber("Median Filter 3D Radius (Z):", 1);
Dialog.addNumber("Target Tile Width (pixels):", 256);
Dialog.addNumber("Target Tile Height (pixels):", 256);
Dialog.addChoice("Perform Bleach Correction?", newArray("Yes", "No"));
Dialog.addChoice("Channel to Process (for Multi-Channel Stacks):", newArray("Split and Process All", "Channel 1 Only", "Channel 2 Only", "Channel 3 Only"));
Dialog.show();

// Retrieve User Parameters
rollingBall = Dialog.getNumber();
medianXY = Dialog.getNumber();
medianZ = Dialog.getNumber();
tileWidth = Dialog.getNumber();
tileHeight = Dialog.getNumber();
doBleachCorr = Dialog.getChoice();
channelMode = Dialog.getChoice();

// Select Input and Output Directories
inputDir = getDirectory("Choose the Input Directory containing WKY/SHR Z-stacks");
outputDir = getDirectory("Choose the Output Directory to save the Ilastik-ready tiles");

setBatchMode(true); // Run in headless/batch mode for speed

fileList = getFileList(inputDir);
for (i = 0; i < fileList.length; i++) {
    fileName = fileList[i];
    
    // Process only TIFF or CZI files
    if (endsWith(fileName, ".tif") || fileName.endsWith(".tiff") || fileName.endsWith(".czi")) {
        showProgress(i + 1, fileList.length);
        print("Processing: " + fileName);
        
        open(inputDir + fileName);
        originalId = getImageID();
        baseName = File.nameWithoutExtension;
        
        // Ensure image is a 3D Z-stack
        getDimensions(width, height, channels, slices, frames);
        if (slices <= 1 && frames > 1) {
            // Fix common dimension interchanging issue (2D+Time vs 3D Stack)
            run("Properties...", "channels=" + channels + " slices=" + frames + " frames=" + slices + " unit=pixel");
            getDimensions(width, height, channels, slices, frames);
        }
        
        if (slices <= 1) {
            print("Skipping " + fileName + ": Not a 3D Stack (Slices = " + slices + ").");
            close();
            continue;
        }

        // --- STEP 1: CHANNEL HANDLING ---
        if (channels > 1) {
            if (channelMode == "Channel 1 Only") {
                run("Duplicate...", "duplicate channels=1");
                selectImage(originalId); close();
                originalId = getImageID();
            } else if (channelMode == "Channel 2 Only") {
                run("Duplicate...", "duplicate channels=2");
                selectImage(originalId); close();
                originalId = getImageID();
            } else if (channelMode == "Channel 3 Only") {
                run("Duplicate...", "duplicate channels=3");
                selectImage(originalId); close();
                originalId = getImageID();
            } else {
                // Split all channels and process them sequentially
                run("Split Channels");
                channelList = getList("image.titles");
                for (c = 0; c < channelList.length; c++) {
                    selectImage(channelList[c]);
                    subChanId = getImageID();
                    processStack(subChanId, baseName + "_Ch" + (c+1), outputDir, rollingBall, medianXY, medianZ, tileWidth, tileHeight, doBleachCorr, width, height, slices);
                }
                continue; // Skip the default processing since Split handled it
            }
        }
        
        // Process single or isolated channel
        getDimensions(width, height, channels, slices, frames);
        processStack(originalId, baseName, outputDir, rollingBall, medianXY, medianZ, tileWidth, tileHeight, doBleachCorr, width, height, slices);
    }
}

setBatchMode(false);
print("--- Pre-processing batch complete! All files ready for Ilastik ---");

// Core Processing Pipeline Function
function processStack(imgId, baseName, outDir, rBall, medXY, medZ, tW, tH, bCorr, w, h, s) {
    selectImage(imgId);
    
    // Convert to 16-bit or 8-bit to ensure feature space consistency
    bitD = bitDepth();
    if (bitD == 24) {
        run("8-bit");
    }
    
    // --- STEP 2: BLEACH & AXIAL DECAY CORRECTION ---
    if (bCorr == "Yes") {
        print("   Performing Bleach Correction (Histogram Matching)...");
        run("Bleach Correction", "recorrection=[Histogram Matching]");
        correctedId = getImageID();
        
        // Close old uncorrected image
        selectImage(imgId); close();
        selectImage(correctedId);
        imgId = getImageID();
    }
    
    // --- STEP 3: BACKGROUND SUBTRACTION & DENOISING ---
    print("   Removing background haze and camera noise...");
    run("Subtract Background...", "rolling=" + rBall + " stack");
    
    // Apply 3D median filter to smooth local voxel inhomogeneity while preserving boundaries
    run("Median 3D...", "x=" + medXY + " y=" + medXY + " z=" + medZ);
    
    // --- STEP 4 & 5: TILING, NORMALIZATION & SAVE ---
    print("   Cropping stack into optimized tiles (" + tW + "x" + tH + ")...");
    
    for (y = 0; y < h; y += tH) {
        for (x = 0; x < w; x += tW) {
            
            // Calculate bounds to prevent out-of-bounds cropping
            xEnd = x + tW;
            yEnd = y + tH;
            if (xEnd > w) xEnd = w;
            if (yEnd > h) yEnd = h;
            
            actualW = xEnd - x;
            actualH = yEnd - y;
            
            // Navigate back to processed main image
            selectImage(imgId);
            
            // Specify crop ROI and duplicate the entire Z-depth
            run("Specify...", "width=" + actualW + " height=" + actualH + " x=" + x + " y=" + y + " slice=1 stack");
            run("Duplicate...", "duplicate");
            tileId = getImageID();
            
            // Intensity Standardization & Histogram Normalization per tile 
            // This ensures maximum local contrast and uniform feature space across SHR and WKY
            run("Enhance Contrast", "saturated=0.35 normalize process_all");
            
            // Save current tile
            tileName = baseName + "_tile_x" + x + "_y" + y + ".tif";
            saveAs("Tiff", outDir + File.separator + tileName);
            close(); // Close tile
        }
    }
    
    // Close the processed main image to free up system memory
    selectImage(imgId);
    close();
}
