library(lhs)
n = 1000
d = 6
X = maximinLHS(n, d)
write.csv(X, paste('data/lhs_', toString(n), '.csv', sep=''))
