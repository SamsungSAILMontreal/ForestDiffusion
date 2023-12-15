install.packages('matrixStats')
install.packages('xtable')
install.packages('plotrix')
library(matrixStats)
library(xtable)
library(plotrix)

all_vars = c('iris', 'wine', 'parkinsons', 'climate_model_crashes', 'concrete_compression', 'yacht_hydrodynamics', 'airfoil_self_noise', 'connectionist_bench_sonar', 'ionosphere', 'qsar_biodegradation', 'seeds', 'glass', 'yeast', 'libras', 'planning_relax', 'blood_transfusion', 'breast_cancer_diagnostic', 'connectionist_bench_vowel', 'concrete_slump', 'wine_quality_red', 'wine_quality_white', 'california', 'bean', 'car','congress','tictactoe')
W_vars = c('iris', 'wine', 'parkinsons', 'climate_model_crashes', 'concrete_compression', 'yacht_hydrodynamics', 'airfoil_self_noise', 'connectionist_bench_sonar', 'ionosphere', 'qsar_biodegradation', 'seeds', 'glass', 'yeast', 'libras', 'planning_relax', 'blood_transfusion', 'breast_cancer_diagnostic', 'connectionist_bench_vowel', 'concrete_slump', 'wine_quality_red', 'wine_quality_white', 'car','congress','tictactoe')
R2_vars = c('concrete_compression', 'yacht_hydrodynamics', 'airfoil_self_noise', 'wine_quality_red', 'wine_quality_white', 'california')
F1_vars = c('iris', 'wine', 'parkinsons', 'climate_model_crashes', 'connectionist_bench_sonar', 'ionosphere', 'qsar_biodegradation', 'seeds', 'glass', 'yeast', 'libras', 'planning_relax', 'blood_transfusion', 'breast_cancer_diagnostic', 'connectionist_bench_vowel', 'bean', 'car','congress','tictactoe')

methods = c('GaussianCopula', 'TVAE', 'CTGAN', 'CTABGAN', 'stasy', 'TabDDPM', 'forest_diffusion_vp_nt51_ycond', 'forest_diffusion_flow_nt51_ycond')
methods_all = c('GaussianCopula', 'TVAE', 'CTGAN', 'CTABGAN', 'stasy', 'TabDDPM', 'forest_diffusion_vp_nt51_ycond', 'forest_diffusion_flow_nt51_ycond', 'oracle')

# Change the folder to your own
data = read.csv("C:/Users/Alexia-Mini/Sync/Ubuntu_ML/projects/Tabular_Flow_Matching/Results/Results_tabular - generation_miss.csv")
data = read.csv("C:/Users/alexi/Sync/Ubuntu_ML/projects/Tabular_Flow_Matching/Results/Results_tabular - generation_miss.csv")

# Better scaling for clean tables
data$percent_bias = data$percent_bias / 100 

#### CHOOOSE HERE

#data = data[data$missingness!='MCAR(0.2 MissForest)',] # miceforest results
data = data[data$missingness!='MCAR(0.2 miceforest)',] # MissForest results

####

data_summary = data.frame(matrix(ncol = 9, nrow = 10))
colnames(data_summary) = methods_all
rownames(data_summary) = c("W_train", "W_test", "coverage_train", "coverage_test", "R2_fake", "F1_fake", "class_score", "percent_bias", "coverage_rate", "time")

for (method in methods_all){
	W_train_mean = mean(data$W_train[data$method==method & data$dataset %in% W_vars])
	W_train_sd = std.error(data$W_train[data$method==method & data$dataset %in% W_vars])

	W_test_mean = mean(data$W_test[data$method==method & data$dataset %in% W_vars])
	W_test_sd = std.error(data$W_test[data$method==method & data$dataset %in% W_vars])

	coverage_mean = mean(data$coverage_train[data$method==method & data$dataset %in% all_vars])
	coverage_sd = std.error(data$coverage_train[data$method==method & data$dataset %in% all_vars])

	coverage_test_mean = mean(data$coverage_test[data$method==method & data$dataset %in% all_vars])
	coverage_test_sd = std.error(data$coverage_test[data$method==method & data$dataset %in% all_vars])

	R2_fake_mean = mean(data$R2_fake[data$method==method & data$dataset %in% R2_vars])
	R2_fake_sd = std.error(data$R2_fake[data$method==method & data$dataset %in% R2_vars])

	F1_fake_mean = mean(data$F1_fake[data$method==method & data$dataset %in% F1_vars])
	F1_fake_sd = std.error(data$F1_fake[data$method==method & data$dataset %in% F1_vars])

	percent_bias_mean = mean(data$percent_bias[data$method==method & data$dataset %in% R2_vars])
	percent_bias_sd = std.error(data$percent_bias[data$method==method & data$dataset %in% R2_vars])

	coverage_rate_mean = mean(data$coverage_rate[data$method==method & data$dataset %in% R2_vars])
	coverage_rate_sd = std.error(data$coverage_rate[data$method==method & data$dataset %in% R2_vars])

	class_score_mean = mean(data$class_score[data$method==method & data$dataset %in% all_vars])
	class_score_sd = std.error(data$class_score[data$method==method & data$dataset %in% all_vars])

	time_mean = mean(data$time[data$method==method & data$dataset %in% all_vars])
	time_sd = std.error(data$time[data$method==method & data$dataset %in% all_vars])

	data_summary[method] = c(
		paste0(as.character(round(W_train_mean, 2)), ' (', as.character(round(W_train_sd, 2)), ')'),
		paste0(as.character(round(W_test_mean, 2)), ' (', as.character(round(W_test_sd, 2)), ')'),
		paste0(as.character(round(coverage_mean, 2)), ' (', as.character(round(coverage_sd, 2)), ')'),
		paste0(as.character(round(coverage_test_mean, 2)), ' (', as.character(round(coverage_test_sd, 2)), ')'),
		paste0(as.character(round(R2_fake_mean, 2)), ' (', as.character(round(R2_fake_sd, 2)), ')'),
		paste0(as.character(round(F1_fake_mean, 2)), ' (', as.character(round(F1_fake_sd, 2)), ')'),
		paste0(as.character(round(class_score_mean, 2)), ' (', as.character(round(class_score_sd, 2)), ')'),
		paste0(as.character(round(percent_bias_mean, 2)), ' (', as.character(round(percent_bias_sd, 2)), ')'),
		paste0(as.character(round(coverage_rate_mean, 2)), ' (', as.character(round(coverage_rate_sd, 2)), ')'),
		paste0(as.character(round(time_mean, 2)), ' (', as.character(round(time_sd, 2)), ')')
		)

}
data_summary
data_summary_t <- t(data_summary)
data_summary_t



###################### RANK

data_summary_rank = data.frame(matrix(ncol = 8, nrow = 10))
colnames(data_summary_rank) = methods
rownames(data_summary) = c("W_train", "W_test", "coverage_train", "coverage_test", "R2_fake", "F1_fake", "class_score", "percent_bias", "coverage_rate", "time")

W_train = data.frame(matrix(ncol = 8, nrow = length(W_vars)))
colnames(W_train) = methods
rownames(W_train) = W_vars
W_test = data.frame(matrix(ncol = 8, nrow = length(W_vars)))
colnames(W_test) = methods
rownames(W_test) = W_vars
coverage = data.frame(matrix(ncol = 8, nrow = length(all_vars)))
colnames(coverage) = methods
rownames(coverage) = all_vars
coverage_test = data.frame(matrix(ncol = 8, nrow = length(all_vars)))
colnames(coverage_test) = methods
rownames(coverage_test) = all_vars
R2_fake = data.frame(matrix(ncol = 8, nrow = length(R2_vars)))
colnames(R2_fake) = methods
rownames(R2_fake) = R2_vars
F1_fake = data.frame(matrix(ncol = 8, nrow = length(F1_vars)))
colnames(F1_fake) = methods
rownames(F1_fake) = F1_vars
percent_bias = data.frame(matrix(ncol = 8, nrow = length(R2_vars)))
colnames(percent_bias) = methods
rownames(percent_bias) = R2_vars
coverage_rate = data.frame(matrix(ncol = 8, nrow = length(R2_vars)))
colnames(coverage_rate) = methods
rownames(coverage_rate) = R2_vars
ClassScore = data.frame(matrix(ncol = 8, nrow = length(all_vars)))
colnames(ClassScore) = methods
rownames(ClassScore) = all_vars
Time = data.frame(matrix(ncol = 8, nrow = length(all_vars)))
colnames(Time) = methods
rownames(Time) = all_vars

for (method in methods){

	W_train[method] = data$W_train[data$method==method & data$dataset %in% W_vars]
	W_test[method] = data$W_test[data$method==method & data$dataset %in% W_vars]

	coverage[method] = -data$coverage_train[data$method==method & data$dataset %in% all_vars]
	coverage_test[method] = -data$coverage_test[data$method==method & data$dataset %in% all_vars]

	R2_fake[method] = -data$R2_fake[data$method==method & data$dataset %in% R2_vars]
	F1_fake[method] = -data$F1_fake[data$method==method & data$dataset %in% F1_vars]
	percent_bias[method] = data$percent_bias[data$method==method & data$dataset %in% R2_vars]
	coverage_rate[method] = -data$coverage_rate[data$method==method & data$dataset %in% R2_vars]
	ClassScore[method] = data$class_score[data$method==method & data$dataset %in% all_vars]
	Time[method] = data$time[data$method==method & data$dataset %in% all_vars]

}

# Rank by dataset

W_train_ = as.data.frame(t(sapply(as.data.frame(t(W_train)), rank)))
colnames(W_train_) = colnames(W_train)
W_train = W_train_

W_test_ = as.data.frame(t(sapply(as.data.frame(t(W_test)), rank)))
colnames(W_test_) = colnames(W_test)
W_test = W_test_

coverage_ = as.data.frame(t(sapply(as.data.frame(t(coverage)), rank)))
colnames(coverage_) = colnames(coverage)
coverage = coverage_
coverage_test_ = as.data.frame(t(sapply(as.data.frame(t(coverage_test)), rank)))
colnames(coverage_test_) = colnames(coverage_test)
coverage_test = coverage_test_

R2_fake_ = as.data.frame(t(sapply(as.data.frame(t(R2_fake)), rank)))
colnames(R2_fake_) = colnames(R2_fake)
R2_fake = R2_fake_

F1_fake_ = as.data.frame(t(sapply(as.data.frame(t(F1_fake)), rank)))
colnames(F1_fake_) = colnames(F1_fake)
F1_fake = F1_fake_

percent_bias_ = as.data.frame(t(sapply(as.data.frame(t(percent_bias)), rank)))
colnames(percent_bias_) = colnames(percent_bias)
percent_bias = percent_bias_
coverage_rate_ = as.data.frame(t(sapply(as.data.frame(t(coverage_rate)), rank)))
colnames(coverage_rate_) = colnames(coverage_rate)
coverage_rate = coverage_rate_

ClassScore_ = as.data.frame(t(sapply(as.data.frame(t(ClassScore)), rank)))
colnames(ClassScore_) = colnames(ClassScore)
ClassScore = ClassScore_
Time_ = as.data.frame(t(sapply(as.data.frame(t(Time)), rank)))
colnames(Time_) = colnames(Time)
Time = Time_

for (method in methods){
	W_train_mean = mean(unlist(W_train[method]))
	W_train_sd = std.error(unlist(W_train[method]))

	W_test_mean = mean(unlist(W_test[method]))
	W_test_sd = std.error(unlist(W_test[method]))

	coverage_mean = mean(unlist(coverage[method]))
	coverage_sd = std.error(unlist(coverage[method]))
	coverage_test_mean = mean(unlist(coverage_test[method]))
	coverage_test_sd = std.error(unlist(coverage_test[method]))

	R2_fake_mean = mean(unlist(R2_fake[method]))
	R2_fake_sd = std.error(unlist(R2_fake[method]))

	F1_fake_mean = mean(unlist(F1_fake[method]))
	F1_fake_sd = std.error(unlist(F1_fake[method]))

	percent_bias_mean = mean(unlist(percent_bias[method]))
	percent_bias_sd = std.error(unlist(percent_bias[method]))

	coverage_rate_mean = mean(unlist(coverage_rate[method]))
	coverage_rate_sd = std.error(unlist(coverage_rate[method]))

	class_score_mean = mean(unlist(ClassScore[method]))
	class_score_sd = std.error(unlist(ClassScore[method]))

	time_mean = mean(unlist(Time[method]))
	time_sd = std.error(unlist(Time[method]))

	data_summary_rank[method] = c(
		paste0(as.character(round(W_train_mean, 1)), ' (', as.character(round(W_train_sd, 1)), ')'),
		paste0(as.character(round(W_test_mean, 1)), ' (', as.character(round(W_test_sd, 1)), ')'),
		paste0(as.character(round(coverage_mean, 1)), ' (', as.character(round(coverage_sd, 1)), ')'),
		paste0(as.character(round(coverage_test_mean, 1)), ' (', as.character(round(coverage_test_sd, 1)), ')'),
		paste0(as.character(round(R2_fake_mean, 1)), ' (', as.character(round(R2_fake_sd, 1)), ')'),
		paste0(as.character(round(F1_fake_mean, 1)), ' (', as.character(round(F1_fake_sd, 1)), ')'),
		paste0(as.character(round(class_score_mean, 1)), ' (', as.character(round(class_score_sd, 1)), ')'),
		paste0(as.character(round(percent_bias_mean, 1)), ' (', as.character(round(percent_bias_sd, 1)), ')'),
		paste0(as.character(round(coverage_rate_mean, 1)), ' (', as.character(round(coverage_rate_sd, 1)), ')'),
		paste0(as.character(round(time_mean, 1)), ' (', as.character(round(time_sd, 1)), ')')
		)

}
rownames(data_summary_rank) = c("W_train", "W_test", "coverage_train", "coverage_test", "R2_fake", "F1_fake", "class_score", "percent_bias", "coverage_rate", "time")
data_summary_rank
data_summary_rank_t <- t(data_summary_rank)
data_summary_rank_t


########### Latex tables

rownames(data_summary_t) = c("GaussianCopula", "TVAE", "CTGAN", "CTABGAN", "Stasy", "TabDDPM", "Forest-VP", "Forest-Flow", "Oracle")
rownames(data_summary_rank_t) = c("GaussianCopula", "TVAE", "CTGAN", "CTABGAN", "Stasy", "TabDDPM", "Forest-VP", "Forest-Flow")

xtable(data_summary_t[,-10], type = "latex")
xtable(data_summary_rank_t[,-10], type = "latex")


# For building the barplot, to csv

W_train = data.frame(matrix(ncol = 9, nrow = length(all_vars)))
colnames(W_train) = methods_all
rownames(W_train) = all_vars
W_train[,] = 0
W_test = data.frame(matrix(ncol = 9, nrow = length(all_vars)))
colnames(W_test) = methods_all
rownames(W_test) = all_vars
W_test[,] = 0
coverage = data.frame(matrix(ncol = 9, nrow = length(all_vars)))
colnames(coverage) = methods_all
rownames(coverage) = all_vars
coverage[,] = 0
coverage_test = data.frame(matrix(ncol = 9, nrow = length(all_vars)))
colnames(coverage_test) = methods_all
rownames(coverage_test) = all_vars
coverage_test[,] = 0
R2_fake = data.frame(matrix(ncol = 9, nrow = length(all_vars)))
colnames(R2_fake) = methods_all
rownames(R2_fake) = all_vars
R2_fake[,] = 0
F1_fake = data.frame(matrix(ncol = 9, nrow = length(all_vars)))
colnames(F1_fake) = methods_all
rownames(F1_fake) = all_vars
F1_fake[,] = 0
ClassScore = data.frame(matrix(ncol = 9, nrow = length(all_vars)))
colnames(ClassScore) = methods_all
rownames(ClassScore) = all_vars
ClassScore[,] = 0
percent_bias = data.frame(matrix(ncol = 9, nrow = length(all_vars)))
colnames(percent_bias) = methods_all
rownames(percent_bias) = all_vars
percent_bias[,] = 0
coverage_rate = data.frame(matrix(ncol = 9, nrow = length(all_vars)))
colnames(coverage_rate) = methods_all
rownames(coverage_rate) = all_vars
coverage_rate[,] = 0
Time = data.frame(matrix(ncol = 9, nrow = length(all_vars)))
colnames(Time) = methods_all
rownames(Time) = all_vars
Time[,] = 0

for (method in methods_all){
	W_train[rownames(W_train) %in% W_vars, method] = data$W_train[data$method==method & data$dataset %in% W_vars]
	W_test[rownames(W_train) %in% W_vars, method] = data$W_test[data$method==method & data$dataset %in% W_vars]

	coverage[, method] = data$coverage_train[data$method==method & data$dataset %in% all_vars]
	coverage_test[, method] = data$coverage_test[data$method==method & data$dataset %in% all_vars]

	R2_fake[rownames(W_train) %in% R2_vars, method] = data$R2_fake[data$method==method & data$dataset %in% R2_vars]
	F1_fake[rownames(W_train) %in% F1_vars, method] = data$F1_fake[data$method==method & data$dataset %in% F1_vars]
	ClassScore[rownames(W_train) %in% all_vars, method] = data$class_score[data$method==method & data$dataset %in% all_vars]
	percent_bias[rownames(W_train) %in% R2_vars, method] = data$percent_bias[data$method==method & data$dataset %in% R2_vars]
	coverage_rate[rownames(W_train) %in% R2_vars, method] = data$coverage_rate[data$method==method & data$dataset %in% R2_vars]
}

# For building the barplot, to csv
write.csv(W_test, file="gen_W_nmiss.csv")
write.csv(coverage_test, file="gen_cov_nmiss.csv")
write.csv(R2_fake, file="gen_R2_nmiss.csv")
write.csv(F1_fake, file="gen_F1_nmiss.csv")
write.csv(ClassScore, file="gen_disc_nmiss.csv")
write.csv(percent_bias, file="gen_pb_nmiss.csv")
write.csv(coverage_rate, file="gen_cr_nmiss.csv")
