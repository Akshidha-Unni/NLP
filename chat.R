
#read the data
data = read.csv("C:\\Users\\akshi\\Downloads\\archive (1)\\chatgpt.csv",stringsAsFactors = FALSE)

#delete unwanted columns
data = data[,-1]
data = data[1:5000,]

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
w = subset(w, w>=100)
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
freqs = rowSums(data$labels)
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
classifier = randomForest(x = train[-1279],
                          y = train$labels,
                          ntree = 15)
y_pred = predict(classifier, newdata = test[-1279])

accuracy = confusionMatrix(test[,1279], y_pred)$overall["Accuracy"]
table(test$labels,y_pred)
print(accuracy)

#CART
tree = rpart(labels ~ .,data = train, method = "class")

prp(tree)
pred = predict(tree, newdata = test[-1279], type = "class")

acc = confusionMatrix(test[,1279], pred)$overall["Accuracy"]
table(test$labels, pred)
print(acc)

#naivebayes
nb = naiveBayes(x = train[-1279],
                y = train$labels)
nb_pred = predict(nb, newdata = test[-1279])
accur = confusionMatrix(test$labels, nb_pred)$overall["Accuracy"]
table(test$labels, nb_pred)
print(accur)

#c50
install.packages("C50")
library(C50)
cf = C5.0(x = train[-1279],
          y = train$labels)
cf_pred = predict(cf, newdata = test[-1279])

acc_cf = confusionMatrix(test[,1279], cf_pred)$overall["Accuracy"]
table(test$labels,cf_pred)
print(acc_cf)

#decision tree
dc = rpart(formula = labels ~., data = train)
dc_pred = predict(dc, newdata = test[-1279], type = "class")

dc_acc = confusionMatrix(test$labels, dc_pred)$overall["Accuracy"]
table(test$labels, dc_pred)
print(dc_acc)


