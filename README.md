# Shap_Relational_Explanation

# Shap_Relational_Exp

In this project, we are explaining the predictions made by Machine Learning Model using SHAP (https://github.com/slundberg/shap). Here we have tried to represent SHAP relational explanations. The Yelp review data is used in this experiment and the data is available in the link (https://drive.google.com/drive/folders/1o-UmrtdLdYVTvUhWd75khzSqppP1upPq?usp=sharing). Here 2000 reviews (1000 positive reviews and 1000 negative reviews) have been used in the experiment. 

# Run the Code
 1. To get the visualized explanation outcomes in Relational_SHAP_EXPP notebooke run the command Run_program_all().

 2. In order to get explanation accuracy for the both word explanation, and relational explanations respectively in Relational_Shap_Accuracy notebook shap_accuracy() command needs to be run. 
# Run the Code to get the Non-Relational Explanation Accuracy
 In Non_Relational_SHAP_Accuracy notebook SHAP_NON_RELATIONAL_EXPLANATION() command needs to be run.
 
 

# Demo Output 
Review 12 is Truly Predicted: Positive 

SHAP Relational Explanation: '131', '687', '593', '703', '150', '601', '212', '142'. Here '131', '687', '593', '703', '150', '601', '212', and  '142' are related reviews to 12 and have positive contribution to the prediction. 

SHAP Words Explanation: 'enjoy', 'drinks', 'show', 'champagne'. Here  are the word explanations

# Demo Explanation Accuracy
Word_Explanation_Average_Accuracy_Shap

0.278422782037238


Relational_Explanation_Average_Accuracy_Shap

0.5006702073849252

# Demo Non-Relational Explanation Accuracy

 The Explanation Accuracy With Human Feedback: Here we have varied the clusters from, 5,20,35,50, and 75 and compute word explanation accuracy with the increase human feedback along with the increase of clusters. The accuracy is computed considering both posotive and negative reviews, only considerating positive reviews, and only considering negative reviews.

    #Considering Both Positive and Negative Reviews
     Number of Clusters    Explanation Accuracy
     5                      0.3402339181286558
     20                     0.3708722741433033
     35                     0.3791411042944796
     50                     0.39511834319526673
     65                     0.39515669515669494



    #Considering only Positive  Reviews
     Number of Clusters    Explanation Accuracy
            5                 0.38309061488673174
            20                0.4411483253588522
            35                0.44210526315789517
            50                0.46682242990654216
            65                0.4693396226415094
      
      
      
     #Considering only Negative  Reviews
     Number of Clusters    Explanation Accuracy
             5                 0.22848101265822754
             20                0.23973214285714242
             35                0.2666666666666662
             50                0.2713709677419351
             65                0.2820143884892085
            

   # The Explanation Accuracy Without  Human Feedback: Clustering is not applied here. 
       #Considering Both Positive and Negative Reviews
         Explanation Accuracy:  0.29463507625272184
         
         
        #Considering only Positive  Reviews
         Explanation Accuracy:  0.32059925093633085
         
         
         #Considering only Negative  Reviews
         Explanation Accuracy:  0.25852864583333407





The explanation accuracy is computed separately for the word explanations, and relational explanations respectively. The manuall self-annotated (by human) process are considered to generate standard word explanations per query. The nerural network based embedding (Doc2Vec) is used to generate the standard relational explanations per query. Here, the standard explanations are considered as true explanations for the query being truly predicted as a specifice class. We have computed the percentage of SHAP relational model's explanations match with the standard explanations, and represent as explanation accuracy. In order to avoid biasness, we have kept the number of explanations  equal for the both standard explanations and SHAP relational model's explanations while computing explanation accuracy. Specifically, we have used first 5 explanations for the SHAP relational model's explanations based on the shap values, and standard explanations respectively. 

# Demo Visualized Output
# 1. Justification of the Prediction of the review query in terms of graph
The query review is currectly predicted as either positive or negative and from the relational graph as follows it is clear that, review 19 is truly predicted as negative as its related or connected review nodes either Samehotel or Sameuser relation are also negative reviews. 

![image](https://user-images.githubusercontent.com/25291998/126830625-d2cb30d0-09c5-46d3-9d96-ca2c24d1d649.png)


# 2. Sameuser Relation

Here, we have the relational graph where the query 19 predicted as negative review connected to a node with respect to Sameuser relation


![image](https://user-images.githubusercontent.com/25291998/126830658-c9abf7f3-8aac-4a86-adf1-4ad17dff9c21.png)


# 3. Samehotel Relation

Here, we have the relational graph where the query 4 predicted as positive review connected to the nodes with respect to Samehotel relation


![image](https://user-images.githubusercontent.com/25291998/126830694-2ab18a34-21a7-40f9-9898-675cec65f06e.png)

# 4. Word Explanations

Here are the words explanations by SHAP for the review 2 to be truly predicted as positive review:


![image](https://user-images.githubusercontent.com/25291998/126830752-6199cb90-56c3-404b-940f-75eb64e9bf6b.png)








# Packages need to be installed
Python=3.6

Gensim=3.8

Jupyter Notebook
