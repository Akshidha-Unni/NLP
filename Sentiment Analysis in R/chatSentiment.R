
#read the data
data = read.csv("C:\\Users\\akshi\\Downloads\\archive (1)\\chatgpt.csv",stringsAsFactors = FALSE)

#delete unwanted columns
data = data[,-1]
data = data[1:20000,]

#install required packages
install.packages('tm')
install.packages("SnowballC")
install.packages("stringr")
install.packages("caTools")
install.packages("randomForest")
install.packages("caret")
install.packages("rpart")
install.packages("rpart.plot")
install.packages("e1071")
install.packages("wordcloud")

install.packages("tidyverse")    
install.packages("tidytext")    
install.packages("syuzhet")     
install.packages("ggplot2") 

#import required packages
library(tm)
library(SnowballC)
library(stringr)
library(caTools)
library(randomForest)
library(caret)
library(rpart)
library(rpart.plot)
library(e1071)
library(wordcloud)

library(tidyverse)    
library(tidytext)    
library(syuzhet)     
library(ggplot2) 

#create a Vcorpus object 
corpus = VCorpus(VectorSource(data$tweets))

#cleaning the text
#convert to lowercase
corpus = tm_map(corpus,content_transformer(tolower))

#removing links from text
remove_links = function(text) {
  cleaned_text = gsub("(http|https)://[[:alnum:]/\\.]+", "", text)
  return(cleaned_text)
}

for (i in 1:length(corpus)) {
  corpus[[i]] <- content_transformer(remove_links)(corpus[[i]])
}

#removing any special characters 
remove_special_chars <- function(text) {
  cleaned_text <- gsub("[^a-z ]+", "", text)
  return(cleaned_text)
  
}
for (i in 1:length(corpus)) {
  corpus[[i]] <- content_transformer(remove_special_chars)(corpus[[i]])
}

#remove stopwords and perform stemming
corpus = tm_map(corpus,removeWords,stopwords())
corpus = tm_map(corpus,stemDocument)
corpus = tm_map(corpus,stripWhitespace)

#creating the bag of words model
dtm = DocumentTermMatrix(corpus)
dtm = removeSparseTerms(dtm,0.999)


#convert to dataframe
dataset = as.data.frame(as.matrix(dtm))

#barplot
w = colSums(dataset)
w = subset(w, w>=1000)
barplot(w, 
        las = 2,
        col = rainbow(50),
        main = "Frequent Words - Bar plot")

#word cloud 
w = sort(colSums(dataset),decreasing = TRUE)
wordcloud(words = names(w),
          freq = w,
          random.order = FALSE,
          colors = brewer.pal(8,"Dark2"),
          max.words = 100,
          rot.per = 0,
          main = "Word Cloud")

#pie chart
freq = table(data$labels)
pie(freq, labels = names(freq), col = c("skyblue","pink","yellow"),main = "Sentiment Distribution")

#finding associations between three most frequent words
associations = findAssocs(dtm, terms = c("chatgpt","openai","ask"), corlimit = 0.1)
print(associations)

#associations between higher frequency terms
asso = findAssocs(dtm, terms = findFreqTerms(dtm, lowfreq = 300), corlimit = 0.1)
print(asso)

#encoding the labels
data$labels = factor(data$labels,
                     levels = c('neutral','good','bad'),
                     labels = c(1,2,3))
dataset$labels = data$labels

#splitting the dataset into train and test
set.seed(223)
split = sample.split(dataset$labels)
train = subset(dataset, split == TRUE)
test = subset(dataset, split == FALSE)

#random forest classifier
classifier = randomForest(x = train[-1358],
                          y = train$labels,
                          ntree = 15)
y_pred = predict(classifier, newdata = test[-1358])

accuracy = confusionMatrix(test[,1358], y_pred)$overall["Accuracy"]
table(test$labels,y_pred)
print(accuracy)

#CART
tree = rpart(labels ~ .,data = train, method = "class")

prp(tree)
pred = predict(tree, newdata = test[-1358], type = "class")

acc = confusionMatrix(test[,1358], pred)$overall["Accuracy"]
table(test$labels, pred)
print(acc)

#hyperparameter tuning CART model
param_grid <- expand.grid(
  cp = c(0.01, 0.001, 0.0001, 0.1, 0.0000001)  
)

ctrl = trainControl(method = "cv",  # Cross-validation method
                     number = 5)
cart_model = train(x = train[-1358], y  = train$labels,
                    method = "rpart",  # CART model
                    trControl = ctrl,
                    tuneGrid = param_grid)

cp_values = cart_model$results$cp
accuracies = cart_model$results$Accuracy
for (i in 1:length(cp_values)) {
  print(paste("Accuracy for cp =", cp_values[i], ":", accuracies[i]))
}

control = rpart.control(cp = 0.0001)
cart_mod = rpart(labels ~ .,data = train, method = "class", control = control)

prp(cart_mod)
pred = predict(cart_mod, newdata = test[-1358], type = "class")

acc = confusionMatrix(test[,1358], pred)$overall["Accuracy"]
table(test$labels, pred)
print(acc)



