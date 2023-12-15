install.packages('matrixStats')
install.packages('xtable')
library(matrixStats)
library(xtable)

data = read.csv("C:/Users/Alexia-Mini/Sync/Ubuntu_ML/projects/Tabular_Flow_Matching/Results/Results_tabular - imputation.csv")
data = read.csv("C:/Users/alexi/Sync/Ubuntu_ML/projects/Tabular_Flow_Matching/Results/Results_tabular - imputation.csv")

# Better scaling for clean tables
data$PercentBias = data$PercentBias / 100 

# debug small floats
data$MeanVariance[data$MeanVariance < 1e-12] = 0
data$MeanMAD[data$MeanMAD < 1e-12] = 0
data$MedianMAD[data$MedianMAD < 1e-12] = 0

all_vars = c('iris', 'wine', 'parkinsons', 'climate_model_crashes', 'concrete_compression', 'yacht_hydrodynamics', 'airfoil_self_noise', 'connectionist_bench_sonar', 'ionosphere', 'qsar_biodegradation', 'seeds', 'glass', 'yeast', 'libras', 'planning_relax', 'blood_transfusion', 'breast_cancer_diagnostic', 'connectionist_bench_vowel', 'concrete_slump', 'wine_quality_red', 'wine_quality_white', 'california', 'bean', 'car','congress','tictactoe')
W_vars = c('iris', 'wine', 'parkinsons', 'climate_model_crashes', 'concrete_compression', 'yacht_hydrodynamics', 'airfoil_self_noise', 'connectionist_bench_sonar', 'ionosphere', 'qsar_biodegradation', 'seeds', 'glass', 'yeast', 'libras', 'planning_relax', 'blood_transfusion', 'breast_cancer_diagnostic', 'connectionist_bench_vowel', 'concrete_slump', 'wine_quality_red', 'wine_quality_white', 'car','congress','tictactoe')
R2_vars = c('concrete_compression', 'yacht_hydrodynamics', 'airfoil_self_noise', 'wine_quality_red', 'wine_quality_white', 'california')
F1_vars = c('iris', 'wine', 'parkinsons', 'climate_model_crashes', 'connectionist_bench_sonar', 'ionosphere', 'qsar_biodegradation', 'seeds', 'glass', 'yeast', 'libras', 'planning_relax', 'blood_transfusion', 'breast_cancer_diagnostic', 'connectionist_bench_vowel', 'bean', 'car','congress','tictactoe')

methods = c('KNN(n_neighbors=1)', 'ice', 'miceforest', 'MissForest', 'softimpute', 'OT', 'GAIN', 'forest_diffusion_repaint_nt51_ycond')
methods_all = c('KNN(n_neighbors=1)', 'ice', 'miceforest', 'MissForest', 'softimpute', 'OT', 'GAIN', 'forest_diffusion_repaint_nt51_ycond', 'oracle')

data_summary = data.frame(matrix(ncol = length(methods_all), nrow = 10))
colnames(data_summary) = methods_all
rownames(data_summary) = c("MinMAE", "AvgMAE", "W_train", "W_test", "MedianMAD", "R2_imp", "F1_imp", "PercentBias", "CoverageRate", "time")

for (method in methods_all){

	MinMAE_mean = mean(data$MinMAE[data$method==method & data$dataset %in% all_vars])
	MinMAE_sd = std.error(data$MinMAE[data$method==method & data$dataset %in% all_vars])

	AvgMAE_mean = mean(data$AvgMAE[data$method==method & data$dataset %in% all_vars])
	AvgMAE_sd = std.error(data$AvgMAE[data$method==method & data$dataset %in% all_vars])

	W_train_mean = mean(data$W_train[data$method==method & data$dataset %in% W_vars])
	W_train_sd = std.error(data$W_train[data$method==method & data$dataset %in% W_vars])

	W_test_mean = mean(data$W_test[data$method==method & data$dataset %in% W_vars])
	W_test_sd = std.error(data$W_test[data$method==method & data$dataset %in% W_vars])

	PercentBias_mean = mean(data$PercentBias[data$method==method & data$dataset %in% R2_vars])
	PercentBias_sd = std.error(data$PercentBias[data$method==method & data$dataset %in% R2_vars])

	CoverageRate_mean = mean(data$CoverageRate[data$method==method & data$dataset %in% R2_vars])
	CoverageRate_sd = std.error(data$CoverageRate[data$method==method & data$dataset %in% R2_vars])

	MedianMAD_mean = mean(data$MedianMAD[data$method==method & data$dataset %in% all_vars])
	MedianMAD_sd = std.error(data$MedianMAD[data$method==method & data$dataset %in% all_vars])

	R2_imp_mean = mean(data$R2_imp[data$method==method & data$dataset %in% R2_vars])
	R2_imp_sd = std.error(data$R2_imp[data$method==method & data$dataset %in% R2_vars])

	F1_imp_mean = mean(data$F1_imp[data$method==method & data$dataset %in% F1_vars])
	F1_imp_sd = std.error(data$F1_imp[data$method==method & data$dataset %in% F1_vars])

	time_mean = mean(data$time[data$method==method & data$dataset %in% all_vars])
	time_sd = std.error(data$time[data$method==method & data$dataset %in% all_vars])

	data_summary[method] = c(
		paste0(as.character(round(MinMAE_mean, 2)), ' (', as.character(round(MinMAE_sd, 2)), ')'),
		paste0(as.character(round(AvgMAE_mean, 2)), ' (', as.character(round(AvgMAE_sd, 2)), ')'),
		paste0(as.character(round(W_train_mean, 2)), ' (', as.character(round(W_train_sd, 2)), ')'),
		paste0(as.character(round(W_test_mean, 2)), ' (', as.character(round(W_test_sd, 2)), ')'),
		paste0(as.character(round(MedianMAD_mean, 2)), ' (', as.character(round(MedianMAD_sd, 2)), ')'),
		paste0(as.character(round(R2_imp_mean, 2)), ' (', as.character(round(R2_imp_sd, 2)), ')'),
		paste0(as.character(round(F1_imp_mean, 2)), ' (', as.character(round(F1_imp_sd, 2)), ')'),
		paste0(as.character(round(PercentBias_mean, 2)), ' (', as.character(round(PercentBias_sd, 2)), ')'),
		paste0(as.character(round(CoverageRate_mean, 2)), ' (', as.character(round(CoverageRate_sd, 2)), ')'),
		paste0(as.character(round(time_mean, 2)), ' (', as.character(round(time_sd, 2)), ')')
		)

}
data_summary
data_summary_t <- t(data_summary)
data_summary_t

###################### RANK

data_summary_rank = data.frame(matrix(ncol = length(methods), nrow = 10))
colnames(data_summary_rank) = methods
rownames(data_summary_rank) = c("MinMAE", "AvgMAE", "W_train", "W_test", "MedianMAD", "R2_imp", "F1_imp", "PercentBias", "CoverageRate", "time")

MinMAE = data.frame(matrix(ncol = length(methods), nrow = length(all_vars)))
colnames(MinMAE) = methods
rownames(MinMAE) = all_vars
AvgMAE = data.frame(matrix(ncol = length(methods), nrow = length(all_vars)))
colnames(AvgMAE) = methods
rownames(AvgMAE) = all_vars
W_train = data.frame(matrix(ncol = length(methods), nrow = length(W_vars)))
colnames(W_train) = methods
rownames(W_train) = W_vars
W_test = data.frame(matrix(ncol = length(methods), nrow = length(W_vars)))
colnames(W_test) = methods
rownames(W_test) = W_vars
MedianMAD = data.frame(matrix(ncol = length(methods), nrow = length(all_vars)))
colnames(MedianMAD) = methods
rownames(MedianMAD) = all_vars
R2_imp = data.frame(matrix(ncol = length(methods), nrow = length(R2_vars)))
colnames(R2_imp) = methods
rownames(R2_imp) = R2_vars
F1_imp = data.frame(matrix(ncol = length(methods), nrow = length(F1_vars)))
colnames(F1_imp) = methods
rownames(F1_imp) = F1_vars
PercentBias = data.frame(matrix(ncol = length(methods), nrow = length(R2_vars)))
colnames(PercentBias) = methods
rownames(PercentBias) = R2_vars
CoverageRate = data.frame(matrix(ncol = length(methods), nrow = length(R2_vars)))
colnames(CoverageRate) = methods
rownames(CoverageRate) = R2_vars
Time = data.frame(matrix(ncol = length(methods), nrow = length(all_vars)))
colnames(Time) = methods
rownames(Time) = all_vars

for (method in methods){
	print(method)
	MinMAE[method] = data$MinMAE[data$method==method & data$dataset %in% all_vars]
	AvgMAE[method] = data$AvgMAE[data$method==method & data$dataset %in% all_vars]
	print(method)
	W_train[method] = data$W_train[data$method==method & data$dataset %in% W_vars]
	W_test[method] = data$W_test[data$method==method & data$dataset %in% W_vars]
	print(method)
	MedianMAD[method] = -data$MedianMAD[data$method==method & data$dataset %in% all_vars]
	print(method)
	PercentBias[method] = data$PercentBias[data$method==method & data$dataset %in% R2_vars]
	CoverageRate[method] = -data$CoverageRate[data$method==method & data$dataset %in% R2_vars]
	print(method)
	R2_imp[method] = -data$R2_imp[data$method==method & data$dataset %in% R2_vars]
	F1_imp[method] = -data$F1_imp[data$method==method & data$dataset %in% F1_vars]

	Time[method] = data$time[data$method==method & data$dataset %in% all_vars]

}

# Rank by dataset

MinMAE_ = as.data.frame(t(sapply(as.data.frame(t(MinMAE)), rank)))
colnames(MinMAE_) = colnames(MinMAE)
MinMAE = MinMAE_

AvgMAE_ = as.data.frame(t(sapply(as.data.frame(t(AvgMAE)), rank)))
colnames(AvgMAE_) = colnames(AvgMAE)
AvgMAE = AvgMAE_

W_train_ = as.data.frame(t(sapply(as.data.frame(t(W_train)), rank)))
colnames(W_train_) = colnames(W_train)
W_train = W_train_

W_test_ = as.data.frame(t(sapply(as.data.frame(t(W_test)), rank)))
colnames(W_test_) = colnames(W_test)
W_test = W_test_

MedianMAD_ = as.data.frame(t(sapply(as.data.frame(t(MedianMAD)), rank)))
colnames(MedianMAD_) = colnames(MedianMAD)
MedianMAD = MedianMAD_

PercentBias_ = as.data.frame(t(sapply(as.data.frame(t(PercentBias)), rank)))
colnames(PercentBias_) = colnames(PercentBias)
PercentBias = PercentBias_

CoverageRate_ = as.data.frame(t(sapply(as.data.frame(t(CoverageRate)), rank)))
colnames(CoverageRate_) = colnames(CoverageRate)
CoverageRate = CoverageRate_

R2_imp_ = as.data.frame(t(sapply(as.data.frame(t(R2_imp)), rank)))
colnames(R2_imp_) = colnames(R2_imp)
R2_imp = R2_imp_

F1_imp_ = as.data.frame(t(sapply(as.data.frame(t(F1_imp)), rank)))
colnames(F1_imp_) = colnames(F1_imp)
F1_imp = F1_imp_

Time_ = as.data.frame(t(sapply(as.data.frame(t(Time)), rank)))
colnames(Time_) = colnames(Time)
Time = Time_

for (method in methods){

	AvgMAE_mean = mean(unlist(AvgMAE[method]))
	AvgMAE_sd = std.error(unlist(AvgMAE[method]))

	MinMAE_mean = mean(unlist(MinMAE[method]))
	MinMAE_sd = std.error(unlist(MinMAE[method]))

	W_train_mean = mean(unlist(W_train[method]))
	W_train_sd = std.error(unlist(W_train[method]))

	W_test_mean = mean(unlist(W_test[method]))
	W_test_sd = std.error(unlist(W_test[method]))

	MedianMAD_mean = mean(unlist(MedianMAD[method]))
	MedianMAD_sd = std.error(unlist(MedianMAD[method]))

	PercentBias_mean = mean(unlist(PercentBias[method]))
	PercentBias_sd = std.error(unlist(PercentBias[method]))

	CoverageRate_mean = mean(unlist(CoverageRate[method]))
	CoverageRate_sd = std.error(unlist(CoverageRate[method]))

	R2_imp_mean = mean(unlist(R2_imp[method]))
	R2_imp_sd = std.error(unlist(R2_imp[method]))

	F1_imp_mean = mean(unlist(F1_imp[method]))
	F1_imp_sd = std.error(unlist(F1_imp[method]))

	time_mean = mean(unlist(Time[method]))
	time_sd = std.error(unlist(Time[method]))

	data_summary_rank[method] = c(
		paste0(as.character(round(AvgMAE_mean, 1)), ' (', as.character(round(AvgMAE_sd, 1)), ')'),
		paste0(as.character(round(MinMAE_mean, 1)), ' (', as.character(round(MinMAE_sd, 1)), ')'),
		paste0(as.character(round(W_train_mean, 1)), ' (', as.character(round(W_train_sd, 1)), ')'),
		paste0(as.character(round(W_test_mean, 1)), ' (', as.character(round(W_test_sd, 1)), ')'),
		paste0(as.character(round(MedianMAD_mean, 1)), ' (', as.character(round(MedianMAD_sd, 1)), ')'),
		paste0(as.character(round(PercentBias_mean, 1)), ' (', as.character(round(PercentBias_sd, 1)), ')'),
		paste0(as.character(round(CoverageRate_mean, 1)), ' (', as.character(round(CoverageRate_sd, 1)), ')'),
		paste0(as.character(round(R2_imp_mean, 1)), ' (', as.character(round(R2_imp_sd, 1)), ')'),
		paste0(as.character(round(F1_imp_mean, 1)), ' (', as.character(round(F1_imp_sd, 1)), ')'),
		paste0(as.character(round(time_mean, 1)), ' (', as.character(round(time_sd, 1)), ')')
		)

}
data_summary_rank
data_summary_rank_t <- t(data_summary_rank)
data_summary_rank_t

########### Latex tables

rownames(data_summary_t) = c("KNN", "ICE", "MICE-Forest", "MissForest", "Softimpute", "OT", "GAIN", "Forest-VP", "Oracle")
rownames(data_summary_rank_t) = c("KNN", "ICE", "MICE-Forest", "MissForest", "Softimpute", "OT", "GAIN", "Forest-VP")

xtable(data_summary_t[,-10], type = "latex")
xtable(data_summary_rank_t[,-10], type = "latex")

# For building the barplot, to csv

MinMAE = data.frame(matrix(ncol = 9, nrow = length(all_vars)))
colnames(MinMAE) = methods_all
rownames(MinMAE) = all_vars
MinMAE[,] = 0
AvgMAE = data.frame(matrix(ncol = 9, nrow = length(all_vars)))
colnames(AvgMAE) = methods_all
rownames(AvgMAE) = all_vars
AvgMAE[,] = 0
W_train = data.frame(matrix(ncol = 9, nrow = length(all_vars)))
colnames(W_train) = methods_all
rownames(W_train) = all_vars
W_train[,] = 0
W_test = data.frame(matrix(ncol = 9, nrow = length(all_vars)))
colnames(W_test) = methods_all
rownames(W_test) = all_vars
W_test[,] = 0
MedianMAD = data.frame(matrix(ncol = 9, nrow = length(all_vars)))
colnames(MedianMAD) = methods_all
rownames(MedianMAD) = all_vars
MedianMAD[,] = 0
R2_imp = data.frame(matrix(ncol = 9, nrow = length(all_vars)))
colnames(R2_imp) = methods_all
rownames(R2_imp) = all_vars
R2_imp[,] = 0
F1_imp = data.frame(matrix(ncol = 9, nrow = length(all_vars)))
colnames(F1_imp) = methods_all
rownames(F1_imp) = all_vars
F1_imp[,] = 0
PercentBias = data.frame(matrix(ncol = 9, nrow = length(all_vars)))
colnames(PercentBias) = methods_all
rownames(PercentBias) = all_vars
PercentBias[,] = 0
CoverageRate = data.frame(matrix(ncol = 9, nrow = length(all_vars)))
colnames(CoverageRate) = methods_all
rownames(CoverageRate) = all_vars
CoverageRate[,] = 0

for (method in methods_all){
	MinMAE[rownames(MinMAE) %in% all_vars, method] = data$MinMAE[data$method==method & data$dataset %in% all_vars]
	AvgMAE[rownames(AvgMAE) %in% all_vars, method] = data$AvgMAE[data$method==method & data$dataset %in% all_vars]

	W_train[rownames(W_train) %in% W_vars, method] = data$W_train[data$method==method & data$dataset %in% W_vars]
	W_test[rownames(W_test) %in% W_vars, method] = data$W_test[data$method==method & data$dataset %in% W_vars]
	MedianMAD[, method] = data$MedianMAD[data$method==method & data$dataset %in% all_vars]

	R2_imp[rownames(R2_imp) %in% R2_vars, method] = data$R2_imp[data$method==method & data$dataset %in% R2_vars]
	F1_imp[rownames(F1_imp) %in% F1_vars, method] = data$F1_imp[data$method==method & data$dataset %in% F1_vars]
	PercentBias[rownames(PercentBias) %in% R2_vars, method] = data$PercentBias[data$method==method & data$dataset %in% R2_vars]
	CoverageRate[rownames(CoverageRate) %in% R2_vars, method] = data$CoverageRate[data$method==method & data$dataset %in% R2_vars]
}


# For building the barplot, to csv
write.csv(MinMAE, file="imp_MinMAE.csv")
write.csv(AvgMAE, file="imp_AvgMAE.csv")
write.csv(W_test, file="imp_Wtrain.csv")
write.csv(W_test, file="imp_W.csv")
write.csv(MedianMAD, file="imp_MedianMAD.csv")
write.csv(R2_imp, file="imp_R2.csv")
write.csv(F1_imp, file="imp_F1.csv")
write.csv(PercentBias, file="imp_pb.csv")
write.csv(CoverageRate, file="imp_cr.csv")
