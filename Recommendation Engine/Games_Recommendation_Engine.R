# Problem Statement

# Q) Build a recommender system with the given data using UBCF.

#This dataset is related to the video gaming industry and a survey was conducted to build a 
#recommendation engine so that the store can improve the sales of its gaming DVDs. Snapshot of the dataset is given below. Build a Recommendation Engine and suggest top selling DVDs to the store customers.


# Lets load necessary libraries

library(recommenderlab)
library(reshape2)

data <- read.csv('game.csv',header=TRUE)
data <- data[,2:3]
dim(data)
head(data)

data_matrix <- as.matrix(acast(data,game~rating, fun.aggregate = mean))
dim(data_matrix)


R <- as(data_matrix, 'realRatingMatrix')

rec1 = Recommender(R, method='UBCF')
rec2 = Recommender(R, method= 'IBCF')
#rec3 = Recommender(R, method='SVD')
rec4 = Recommender(R, method='POPULAR')
rec5 = Recommender(binarize(R,minRating=2), method="UBCF") 

id = 'God of War'
games <- subset(data, data$game==id)
games

prediction <- predict(rec1, R[id], n=5)
as(prediction,"list")
prediction <- predict(rec2, R[id], n=3)
as(prediction,"list")
