# Shap_Relational_Explanation

# Shap_Relational_Exp with Review Data

In this project, we are explaining the predictions made by Machine Learning Model using SHAP (https://github.com/slundberg/shap). Here we have tried to represent SHAP relational explanations. The Yelp review data is used in this experiment and the data is available in the link (https://drive.google.com/drive/folders/1o-UmrtdLdYVTvUhWd75khzSqppP1upPq?usp=sharing). Here 2000 reviews (1000 positive reviews and 1000 negative reviews) have been used in the experiment. 

# Run the Code with Review Data
 1. To get the visualized explanation outcomes in Relational_SHAP_EXPP notebooke run the command Run_program_all().

 2. In order to get explanation accuracy for the both word explanation, and relational explanations respectively in Relational_Shap_Accuracy notebook shap_accuracy() command needs to be run. 
# Run the Code to get the Non-Relational Explanation Accuracy with Review Data
 In Non_Relational_SHAP_Accuracy notebook shap_explanation_u,qrat_u=SHAP_NON_RELATIONAL_EXPLANATION() command needs to be run.
 
 # Run the Code for the Tweet Covid Data
 
 In Shap_Corona_Relational_Explanation notebook tweet_covid_relational_explanation() command needs to be run in a cell.
 
 

# Demo Output with Review Data
Review 12 is Truly Predicted: Positive 

SHAP Relational Explanation: '131', '687', '593', '703', '150', '601', '212', '142'. Here '131', '687', '593', '703', '150', '601', '212', and  '142' are related reviews to 12 and have positive contribution to the prediction. 

SHAP Words Explanation: 'enjoy', 'drinks', 'show', 'champagne'. Here  are the word explanations

# Demo  Relational Explanation Accuracy with Review Data
Word_Explanation_Average_Accuracy_Shap

0.278422782037238


Relational_Explanation_Average_Accuracy_Shap

0.5006702073849252

# Demo Relational and Word Explanation Accuracy for the Tweet Covid-19 data
For the Tweet Covid-19 data where the inference is performed using SVM whether the tweet is Scientific and Non-Scientific tweets. Here we have used 667 tweets and SHAP produces both relational explanation and non-relational explanations. The relational explanations are Sameuser relations, it implies that a user wrote multiple tweets. The data used in this case are in csv files. The data can be found in https://drive.google.com/drive/folders/15I5lfiZ5EKLPKvTsJE5fLHDT_7dsWWel

Word_Explanation_Average_Accuracy_Shap

0.5289999999999998


Relational_Explanation_Average_Accuracy_Shap

0.7425742574257421

# Demo Non-Relational Cluster based Human Feedback Explanation Accuracy with Review Data

 The Explanation Accuracy With Human Feedback: Here we have varied the clusters from, 5,20,35,50, and 65 and compute word explanation accuracy with the increase human feedback along with the increase of clusters. The accuracy is computed considering both posotive and negative reviews, only considerating positive reviews, and only considering negative reviews. In this experiment 400 reviews (200 positive, 200 negative) are used

    #Considering Both Positive and Negative Reviews
     Number of Clusters    Explanation Accuracy
     5                    0.5145833333333336
    20                    0.5145833333333336
    35                    0.5098214285714288
    50                    0.5098214285714288
    65                    0.5014880952380953




    #Considering only Positive  Reviews
     Number of Clusters    Explanation Accuracy
              5              0.5471264367816092
             20              0.5471264367816092
             35              0.5379310344827587
             50              0.5379310344827587
             65              0.5218390804597703
      
      
      
     #Considering only Negative  Reviews
     Number of Clusters    Explanation Accuracy
             5           0.47962962962962963
             20          0.47962962962962963
             35          0.47962962962962963
             50          0.47962962962962963
             65           0.47962962962962963
            

   # The Explanation Accuracy Without  Human Feedback with Review Data: Clustering is not applied here. 
       #Considering Both Positive and Negative Reviews
         Explanation Accuracy:  0.29463507625272184
         
         
        #Considering only Positive  Reviews
         Explanation Accuracy:  0.32059925093633085
         
         
         #Considering only Negative  Reviews
         Explanation Accuracy:  0.25852864583333407





The explanation accuracy is computed separately for the word explanations, and relational explanations respectively. The manuall self-annotated (by human) process are considered to generate standard word explanations per query. The nerural network based embedding (Doc2Vec) is used to generate the standard relational explanations per query. Here, the standard explanations are considered as true explanations for the query being truly predicted as a specifice class. We have computed the percentage of SHAP relational model's explanations match with the standard explanations, and represent as explanation accuracy. In order to avoid biasness, we have kept the number of explanations  equal for the both standard explanations and SHAP relational model's explanations while computing explanation accuracy. Specifically, we have used first 5 explanations for the SHAP relational model's explanations based on the shap values, and standard explanations respectively. 

# Demo Visualized Output
# 1. Justification of the Prediction of the review query in terms of graph with Review Data
The query review is currectly predicted as either positive or negative and from the relational graph as follows it is clear that, review 19 is truly predicted as negative as its related or connected review nodes either Samehotel or Sameuser relation are also negative reviews. 

![image](https://user-images.githubusercontent.com/25291998/126830625-d2cb30d0-09c5-46d3-9d96-ca2c24d1d649.png)


# 2. Sameuser Relation with Review Data

Here, we have the relational graph where the query 19 predicted as negative review connected to a node with respect to Sameuser relation


![image](https://user-images.githubusercontent.com/25291998/126830658-c9abf7f3-8aac-4a86-adf1-4ad17dff9c21.png)


# 3. Samehotel Relation with Review Data

Here, we have the relational graph where the query 4 predicted as positive review connected to the nodes with respect to Samehotel relation


![image](https://user-images.githubusercontent.com/25291998/126830694-2ab18a34-21a7-40f9-9898-675cec65f06e.png)

# 4. Word Explanations with Review Data

Here are the words explanations by SHAP for the review 2 to be truly predicted as positive review:


![image](https://user-images.githubusercontent.com/25291998/126830752-6199cb90-56c3-404b-940f-75eb64e9bf6b.png)








# Packages need to be installed
Python=3.6

Gensim=3.8

Jupyter Notebook
