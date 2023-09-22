
# Alexia: These file is for me for manual installation, feel free to ignore this

# Install
setwd("C:/Users/Alexia-Mini/Sync/Ubuntu_ML/Projects/ForestDiffusion_SAIL/R-package") 
require("roxygen2") || install.packages("roxygen2")
require("devtools") || install.packages("devtools")
require("rmarkdown") || install.packages("rmarkdown")
require("knitr") || install.packages("knitr")
require("pandoc") || install.packages("pandoc")
Sys.setenv(RSTUDIO_PANDOC="C:/Program Files/Pandoc")
options(encoding = 'UTF-8')
roxygenize("ForestDiffusion")
install("ForestDiffusion", dependencies=c("Depends", "Imports"), build_vignettes=FALSE)

# Check
check("ForestDiffusion", cran = TRUE, force_suggests = TRUE, run_dont_test = TRUE, vignettes = TRUE)

# Build (also get the vignettes)
build("ForestDiffusion")

# Get pdf documentation
roxygenize("ForestDiffusion")
install("ForestDiffusion")
pack <- "ForestDiffusion"
path <- find.package(pack)
system(paste(shQuote(file.path(R.home("bin"), "R")),"CMD", "Rd2pdf --force", shQuote(path)))
