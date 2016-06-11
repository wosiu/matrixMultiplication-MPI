library(dplyr)
bench = "column sparse_small 1 0m7.824s
column sparse_small 2 0m7.722s
column sparse_big 1 0m17.928s
column sparse_big 2 0m15.594s
column dense_small 1 5m3.596s
column dense_small 2 5m4.231s
column dense_big 1 1m35.084s
column dense_big 2 1m24.809s
inner sparse_small 1 0m8.287s
inner sparse_small 2 0m8.086s
inner sparse_big 1 0m11.826s
inner sparse_big 2 0m10.946s
inner dense_small 1 5m20.993s
inner dense_small 2 5m19.179s
inner dense_big 1 1m22.998s
inner dense_big 2 1m22.084s"

bench = strsplit(bench, "\n")
bench = unlist(bench)
tmp = lapply (bench, function(x) strsplit(x, " "))
tmp = matrix(unlist(tmp), nrow=4)
tmp = t(tmp)
df = data.frame(tmp)
colnames(df) = c("alg", "matrix", "c", "time")

convert = function(timeStr) {
  t = as.numeric(unlist(strsplit(timeStr, "[ms]")))
  t[1] * 60 + t[2]
}

timeSec = sapply(as.character(df[,"time"]), convert )
df[,"time"] = timeSec

p = df %>% group_by(matrix) %>% summarise(alg=list(alg), c=list(c), time=list(time))

plots = c()
for (i in 1:nrow(p)) {
  data = p[i,]
  title = as.character(data[["matrix"]])
  c = data$c[[1]]
  alg = data$alg[[1]]
  time = data$time[[1]]
  
  tmp = data.frame(alg = factor(alg), c = factor(c), time=time)
  g <- ggplot(tmp, aes(c, fill=alg,weight=time)) + geom_bar(position="dodge") + ylab("real time [s]") + ggtitle(title) + xlab("replication factor")
  plots = c(plots, list(g))
}
plot_grid(plotlist = plots, ncol = 2)

