library(ForestDiffusion)

set.seed(1) 

# Load iris
data(iris)
# variables 1 to 4 are the input X
# variable 5 (iris$Species) is the outcome (class with 3 labels)

# Setup data
X = data.frame(iris[,1:4])
y = iris$Species
Xy = iris
plot(Xy)

Xy$cont1 = rnorm(150)

# Add new categorical variable
Xy$cata = 1
Xy$cata[1:25] = 0
Xy$cata = factor(Xy$cata)

Xy$catb = "0"
Xy$catb[1:45] = "1"
Xy$catb[48:70] = "2"
Xy$catb = factor(Xy$catb)

Xy$catc = "0.1"
Xy$catc[1:45] = "0.2"
Xy$catc = factor(Xy$catc)

Xy$cont2 = rnorm(150)

Xy$catd = "1.1"
Xy$catd[1:15] = "1.2"
Xy$catd[48:52] = "1.4"
Xy$catd = factor(Xy$catd)

Xy$cont3 = rnorm(150)

# Add NAs (but not to label) to emulate having a dataset with missing values
Xy = missForest::prodNA(Xy, noNA = 0.1)


forest_model = ForestDiffusion(X=Xy, n_cores=2, n_t=2, duplicate_K=1, flow=TRUE, seed=666)
Xy_fake = ForestDiffusion.generate(forest_model, batch_size=NROW(Xy), seed=3) # breaks in the old code, but works now
plot(Xy_fake)