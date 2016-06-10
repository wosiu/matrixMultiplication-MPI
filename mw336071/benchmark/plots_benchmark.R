library(dplyr)
library(ggplot2)
library(cowplot)
proc_num = 8
header = c("activity", "sparse_mode", "num_processes", "repl_fact", "exponent", "dim", "cells")
header = c(header, paste0("p", as.character(1:proc_num)))
res = read.csv(file("mw336071/benchmark/result.csv"), sep="\t", header = F)
res = res[,-ncol(res)]
colnames(res) = header

mean = apply(res[,8:ncol(res)], 1, mean)
sd = apply(res[,8:ncol(res)], 1, sd)
sd2 = apply(res[,8:ncol(res)], 1, function(v) sd(v[-which.min(v)]) )

stat = cbind(res[,1:7],mean,sd,sd2)

stat = stat %>% select(-num_processes, -exponent)

p = stat %>% group_by(dim,cells,activity) %>% summarise(means=list(mean), sds=list(sd), sd2s=list(sd2), c=list(repl_fact), modes=list(sparse_mode))
library('scales')
plots = c()
for (i in 1:nrow(p)) {
  data = p[i,]
  dim = data[["dim"]]
  cells = data[["cells"]]
  if (data[["activity"]] == "comp")
    activ = "obliczenia"
  else
    activ = "początkowa komunikacja"
  title = paste(activ, "\ndim", dim, "cells", cells, "dens", percent(cells/dim^2))
  c = data$c[[1]]
  sds = data$sds[[1]]
  sd2s = data$sd2s[[1]]
  means = data$means[[1]]
  mode = data$modes[[1]]
  df = data.frame(mode = factor(mode),c = factor(c),mean = means, sd=sds, sd2=sd2s)
  class(df)
  g <- ggplot(df, aes(c, fill=mode,weight=sd2)) + geom_bar(position="dodge") + ylab("mean time [s]") + ggtitle(title) + xlab("replication factor")
  plots = c(plots, list(g))
}
plot_grid(plotlist = plots, ncol = 2)

####################

bench = read.csv(file("mw336071/benchmarki MPI - Arkusz2.csv"), header=F, dec = ",")
bench[,"V5"] = as.factor(bench[,"V5"])
means = apply(bench[,1:4], 1, mean)
bench = bench %>% select( -c(1:4) )
colnames(bench) = c("proc", "method") 
bench = cbind( bench, means )

bench = bench %>% group_by(proc, method) %>% summarise(m = mean(means))
df = bench
ggplot(df, aes(proc, fill=method, weight=m)) + geom_bar(position="dodge") + ylab("mean time [s]") + xlab("proc number")

