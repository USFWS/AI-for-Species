
#library (raster)
library (terra)
#library (rgdal)

# Input: Set working dir for ortho/mosaics
setwd(file.path('D:', 'species_2025', '14_orthos_wgs1984'))


input_dir <- "D:/species_2025/14_orthos_wgs1984/"
output_dir <- "D:/species_2025/14_orthos_utm"
target_crs <- "EPSG:32614" 

raster_files <- list.files (pattern="*tif$")  
raster_files

for (file_path in raster_files) {
  # Read the raster
  r <- rast(file_path)
  
  # Reproject the raster
  # The 'method' argument can be adjusted (e.g., "bilinear", "cubic")
  reprojected_r <- project(r, target_crs, method = "bilinear")
  
  # Construct the output file name
  # Extract original file name without extension
  file_name <- tools::file_path_sans_ext(basename(file_path))
  file_name
  output_file_path <- file.path(output_dir, paste0(file_name, ".tif"))
  output_file_path
  
  # Save the reprojected raster
  writeRaster(reprojected_r, output_file_path, overwrite = TRUE)
}



